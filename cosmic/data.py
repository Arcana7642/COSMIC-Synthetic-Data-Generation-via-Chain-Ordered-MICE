from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str
    is_target: bool
    position: int


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    info: dict[str, Any]
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    specs: list[ColumnSpec]

    @property
    def all_columns(self) -> list[str]:
        return [spec.name for spec in self.specs]

    @property
    def target_columns(self) -> list[str]:
        return [spec.name for spec in self.specs if spec.is_target]

    @property
    def feature_columns(self) -> list[str]:
        return [spec.name for spec in self.specs if not spec.is_target]

    @property
    def numerical_columns(self) -> list[str]:
        return [spec.name for spec in self.specs if spec.kind == "numerical"]

    @property
    def categorical_columns(self) -> list[str]:
        return [spec.name for spec in self.specs if spec.kind == "categorical"]

    @property
    def train_size(self) -> int:
        return len(self.train_df)


def _load_info(dataname: str) -> dict[str, Any]:
    info_path = ROOT_DIR / "data" / dataname / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"Processed dataset info was not found at {info_path}. "
            f"Run process_dataset.py for '{dataname}' first."
        )

    with info_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_dataset_bundle(dataname: str) -> DatasetBundle:
    info = _load_info(dataname)
    data_dir = ROOT_DIR / "data" / dataname

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    column_names = list(info["column_names"])
    train_df = train_df[column_names].copy()
    test_df = test_df[column_names].copy()

    numerical_idx = set(info["num_col_idx"])
    categorical_idx = set(info["cat_col_idx"])
    target_idx = set(info["target_col_idx"])
    task_type = info["task_type"]

    specs: list[ColumnSpec] = []
    for idx, name in enumerate(column_names):
        is_target = idx in target_idx
        is_numerical = idx in numerical_idx or (task_type == "regression" and is_target)
        if not is_numerical and idx not in categorical_idx and not is_target:
            raise ValueError(f"Column '{name}' is not covered by dataset metadata.")
        kind = "numerical" if is_numerical else "categorical"
        specs.append(ColumnSpec(name=name, kind=kind, is_target=is_target, position=idx))

    return DatasetBundle(
        name=dataname,
        info=info,
        train_df=train_df,
        test_df=test_df,
        specs=specs,
    )


@dataclass
class EncodedTable:
    num: np.ndarray
    cat: np.ndarray


class TabularPreprocessor:
    def __init__(self, bundle: DatasetBundle):
        self.bundle = bundle
        self.specs = bundle.specs
        self.num_columns = [spec.name for spec in self.specs if spec.kind == "numerical"]
        self.cat_columns = [spec.name for spec in self.specs if spec.kind == "categorical"]
        self.num_stats: dict[str, dict[str, float]] = {}
        self.cat_classes: dict[str, list[str]] = {}
        self.cat_to_idx: dict[str, dict[str, int]] = {}
        self.num_positions = [spec.position for spec in self.specs if spec.kind == "numerical"]
        self.cat_positions = [spec.position for spec in self.specs if spec.kind == "categorical"]
        self.num_position_to_index = {
            pos: idx for idx, pos in enumerate(self.num_positions)
        }
        self.cat_position_to_index = {
            pos: idx for idx, pos in enumerate(self.cat_positions)
        }

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        for col in self.num_columns:
            values = pd.to_numeric(frame[col], errors="coerce").astype(float)
            median = float(values.median())
            filled = values.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std())
            if std < 1e-6:
                std = 1.0
            self.num_stats[col] = {
                "mean": mean,
                "std": std,
                "median": median,
                "min": float(filled.min()),
                "max": float(filled.max()),
            }

        for col in self.cat_columns:
            values = frame[col].fillna("nan").astype(str)
            classes = sorted(values.unique().tolist())
            self.cat_classes[col] = classes
            self.cat_to_idx[col] = {value: idx for idx, value in enumerate(classes)}

        return self

    def _encode_num_col(self, series: pd.Series, col: str) -> np.ndarray:
        stats = self.num_stats[col]
        values = pd.to_numeric(series, errors="coerce").astype(float)
        filled = values.fillna(stats["median"])
        standardized = (filled - stats["mean"]) / stats["std"]
        return standardized.to_numpy(dtype=np.float32)

    def _decode_num_col(self, values: np.ndarray, col: str) -> np.ndarray:
        stats = self.num_stats[col]
        restored = values.astype(np.float32) * stats["std"] + stats["mean"]
        restored = np.clip(restored, stats["min"], stats["max"])
        return restored.astype(np.float32)

    def _encode_cat_col(self, series: pd.Series, col: str) -> np.ndarray:
        values = series.fillna("nan").astype(str)
        mapping = self.cat_to_idx[col]
        fallback = mapping["nan"] if "nan" in mapping else 0
        encoded = values.map(lambda value: mapping.get(value, fallback))
        return encoded.to_numpy(dtype=np.int64)

    def _decode_cat_col(self, values: np.ndarray, col: str) -> np.ndarray:
        classes = self.cat_classes[col]
        clipped = np.clip(values.astype(int), 0, len(classes) - 1)
        return np.asarray([classes[idx] for idx in clipped], dtype=object)

    def encode_dataframe(self, frame: pd.DataFrame) -> EncodedTable:
        num = (
            np.stack([self._encode_num_col(frame[col], col) for col in self.num_columns], axis=1)
            if self.num_columns
            else np.empty((len(frame), 0), dtype=np.float32)
        )
        cat = (
            np.stack([self._encode_cat_col(frame[col], col) for col in self.cat_columns], axis=1)
            if self.cat_columns
            else np.empty((len(frame), 0), dtype=np.int64)
        )
        return EncodedTable(num=num, cat=cat)

    def decode_dataframe(self, encoded: EncodedTable) -> pd.DataFrame:
        data: dict[str, np.ndarray] = {}

        for idx, col in enumerate(self.num_columns):
            data[col] = self._decode_num_col(encoded.num[:, idx], col)

        for idx, col in enumerate(self.cat_columns):
            data[col] = self._decode_cat_col(encoded.cat[:, idx], col)

        frame = pd.DataFrame(data)
        return frame[self.bundle.all_columns].copy()

    def to_tensors(self, frame: pd.DataFrame, device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_dataframe(frame)
        num_tensor = torch.from_numpy(encoded.num).to(device=device, dtype=torch.float32)
        cat_tensor = torch.from_numpy(encoded.cat).to(device=device, dtype=torch.long)
        return num_tensor, cat_tensor

    def encode_categorical_target(self, col: str, series: pd.Series) -> np.ndarray:
        return self._encode_cat_col(series, col)

    def decode_categorical_target(self, col: str, values: np.ndarray) -> np.ndarray:
        return self._decode_cat_col(values, col)

    def numeric_bounds(self, col: str) -> tuple[float, float]:
        stats = self.num_stats[col]
        return stats["min"], stats["max"]
