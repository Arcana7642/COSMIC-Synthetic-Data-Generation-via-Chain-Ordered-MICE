from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from cosmic.config import COSMICConfig
from cosmic.data import DatasetBundle, TabularPreprocessor


@dataclass(frozen=True)
class ColumnImportance:
    name: str
    shap_value: float


@dataclass(frozen=True)
class GenerationPlan:
    ordered_columns: list[str]
    importances: list[ColumnImportance]

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "GenerationPlan":
        payload = json.loads(path.read_text(encoding="utf-8"))
        importances = [ColumnImportance(**item) for item in payload["importances"]]
        return cls(ordered_columns=payload["ordered_columns"], importances=importances)


class DummyFeatureMapper:
    def __init__(self, predictor_cols: list[str], categorical_cols: list[str]):
        self.predictor_cols = predictor_cols
        self.categorical_cols = [col for col in predictor_cols if col in categorical_cols]
        self.feature_columns: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "DummyFeatureMapper":
        encoded = pd.get_dummies(
            frame[self.predictor_cols],
            columns=self.categorical_cols,
            prefix_sep="::",
            dtype=float,
        )
        self.feature_columns = encoded.columns.tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        encoded = pd.get_dummies(
            frame[self.predictor_cols],
            columns=self.categorical_cols,
            prefix_sep="::",
            dtype=float,
        )
        encoded = encoded.reindex(columns=self.feature_columns, fill_value=0.0)
        return encoded.to_numpy(dtype=np.float32)


def _aggregate_shap_values(shap_values: Any, feature_names: list[str]) -> dict[str, float]:
    if isinstance(shap_values, list):
        arrays = [np.asarray(value) for value in shap_values]
        values = np.stack(arrays, axis=0)
    else:
        values = np.asarray(getattr(shap_values, "values", shap_values))

    abs_values = np.abs(values)
    feature_axis = None
    for axis, size in enumerate(abs_values.shape):
        if size == len(feature_names):
            feature_axis = axis
            break
    if feature_axis is None:
        raise ValueError("Could not identify the feature axis in SHAP values.")

    reduce_axes = tuple(axis for axis in range(abs_values.ndim) if axis != feature_axis)
    feature_scores = abs_values.mean(axis=reduce_axes)
    column_scores: dict[str, list[float]] = {}
    for feature_name, score in zip(feature_names, feature_scores.tolist()):
        original = feature_name.split("::", 1)[0]
        column_scores.setdefault(original, []).append(float(score))

    return {name: float(np.mean(scores)) for name, scores in column_scores.items()}


def compute_generation_plan(
    bundle: DatasetBundle,
    preprocessor: TabularPreprocessor,
    latent: np.ndarray,
    config: COSMICConfig,
) -> GenerationPlan:
    target_columns = bundle.target_columns
    predictor_columns = [col for col in bundle.all_columns if col not in target_columns]

    if target_columns:
        target_col = target_columns[0]
        mapper = DummyFeatureMapper(predictor_columns, bundle.categorical_columns)
        mapper.fit(bundle.train_df)
        X = mapper.transform(bundle.train_df)

        if target_col in bundle.numerical_columns:
            y = pd.to_numeric(bundle.train_df[target_col], errors="coerce").fillna(0.0).to_numpy()
            model = RandomForestRegressor(
                n_estimators=config.num_trees,
                random_state=config.seed,
                n_jobs=-1,
            )
        else:
            y = preprocessor.encode_categorical_target(target_col, bundle.train_df[target_col])
            model = RandomForestClassifier(
                n_estimators=config.num_trees,
                random_state=config.seed,
                n_jobs=-1,
            )

        model.fit(X, y)
        shap_rows = min(config.shap_num_samples, len(bundle.train_df))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:shap_rows])
        score_map = _aggregate_shap_values(shap_values, mapper.feature_columns)
    else:
        predictor_columns = bundle.all_columns
        mapper = DummyFeatureMapper(predictor_columns, bundle.categorical_columns)
        mapper.fit(bundle.train_df)
        X = mapper.transform(bundle.train_df)
        model = RandomForestRegressor(
            n_estimators=config.num_trees,
            random_state=config.seed,
            n_jobs=-1,
        )
        model.fit(X, latent[:, 0])
        shap_rows = min(config.shap_num_samples, len(bundle.train_df))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:shap_rows])
        score_map = _aggregate_shap_values(shap_values, mapper.feature_columns)

    importances = [
        ColumnImportance(name=col, shap_value=float(score_map.get(col, 0.0)))
        for col in predictor_columns
    ]
    ordered_predictors = [
        item.name for item in sorted(importances, key=lambda item: (-item.shap_value, item.name))
    ]
    ordered_columns = ordered_predictors + [col for col in target_columns if col not in ordered_predictors]

    return GenerationPlan(ordered_columns=ordered_columns, importances=importances)


@dataclass
class ConditionalModel:
    target_col: str
    target_kind: str
    predictor_cols: list[str]
    mapper: DummyFeatureMapper
    model: Any
    residuals: np.ndarray | None
    classes: list[str] | None
    min_value: float | None
    max_value: float | None

    def _augment(self, X: np.ndarray, latent: np.ndarray) -> np.ndarray:
        return np.concatenate([X, latent], axis=1).astype(np.float32)

    def sample(self, frame: pd.DataFrame, latent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        X = self._augment(self.mapper.transform(frame), latent)
        if self.target_kind == "numerical":
            prediction = self.model.predict(X).astype(np.float32)
            if self.residuals is not None and len(self.residuals) > 0:
                noise = rng.choice(self.residuals, size=len(prediction), replace=True).astype(np.float32)
                prediction = prediction + noise
            return np.clip(prediction, self.min_value, self.max_value)

        probabilities = self.model.predict_proba(X)
        sampled = []
        for row in probabilities:
            row = np.asarray(row, dtype=np.float64)
            row = row / row.sum()
            sampled.append(rng.choice(len(row), p=row))
        sampled = np.asarray(sampled, dtype=int)
        assert self.classes is not None
        return np.asarray([self.classes[idx] for idx in sampled], dtype=object)


def fit_conditional_models(
    bundle: DatasetBundle,
    preprocessor: TabularPreprocessor,
    latent: np.ndarray,
    config: COSMICConfig,
    ordered_columns: list[str],
) -> dict[str, ConditionalModel]:
    models: dict[str, ConditionalModel] = {}
    context = latent[:, : min(config.context_dim, latent.shape[1])]

    for target_col in ordered_columns:
        predictor_cols = [col for col in bundle.all_columns if col != target_col]
        mapper = DummyFeatureMapper(predictor_cols, bundle.categorical_columns)
        mapper.fit(bundle.train_df)
        X = mapper.transform(bundle.train_df)
        X = np.concatenate([X, context], axis=1).astype(np.float32)

        if target_col in bundle.numerical_columns:
            y = pd.to_numeric(bundle.train_df[target_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            model = RandomForestRegressor(
                n_estimators=config.num_trees,
                random_state=config.seed,
                n_jobs=-1,
            )
            model.fit(X, y)
            residuals = y - model.predict(X)
            min_value, max_value = preprocessor.numeric_bounds(target_col)
            models[target_col] = ConditionalModel(
                target_col=target_col,
                target_kind="numerical",
                predictor_cols=predictor_cols,
                mapper=mapper,
                model=model,
                residuals=np.asarray(residuals, dtype=np.float32),
                classes=None,
                min_value=min_value,
                max_value=max_value,
            )
        else:
            y = preprocessor.encode_categorical_target(target_col, bundle.train_df[target_col])
            model = RandomForestClassifier(
                n_estimators=config.num_trees,
                random_state=config.seed,
                n_jobs=-1,
            )
            model.fit(X, y)
            models[target_col] = ConditionalModel(
                target_col=target_col,
                target_kind="categorical",
                predictor_cols=predictor_cols,
                mapper=mapper,
                model=model,
                residuals=None,
                classes=preprocessor.cat_classes[target_col],
                min_value=None,
                max_value=None,
            )

    return models
