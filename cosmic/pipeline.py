from __future__ import annotations

import pickle
import random
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cosmic.config import COSMICConfig
from cosmic.data import DatasetBundle, EncodedTable, TabularPreprocessor, load_dataset_bundle
from cosmic.mice import ColumnImportance, ConditionalModel, GenerationPlan, compute_generation_plan, fit_conditional_models
from cosmic.model import TabularMAE, sample_column_mask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class COSMICPipeline:
    def __init__(self, config: COSMICConfig):
        self.config = config
        self.root_dir = Path(__file__).resolve().parents[1]
        self.bundle: DatasetBundle | None = None
        self.preprocessor: TabularPreprocessor | None = None
        self.model: TabularMAE | None = None
        self.artifact_context_dim = config.context_dim

    def describe(self) -> str:
        return (
            "COSMIC pipeline: masked autoencoder training -> column-wise SHAP ranking "
            "-> order-aware MICE generation in descending SHAP order."
        )

    def load_bundle(self) -> DatasetBundle:
        if self.bundle is None:
            self.bundle = load_dataset_bundle(self.config.dataname)
        return self.bundle

    def build_preprocessor(self) -> TabularPreprocessor:
        bundle = self.load_bundle()
        if self.preprocessor is None:
            self.preprocessor = TabularPreprocessor(bundle).fit(bundle.train_df)
        return self.preprocessor

    def build_model(self) -> TabularMAE:
        preprocessor = self.build_preprocessor()
        if self.model is None:
            self.model = TabularMAE(
                preprocessor=preprocessor,
                hidden_dim=self.config.hidden_dim,
                depth=self.config.depth,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
            )
        return self.model

    def _compute_masked_loss(
        self,
        model: TabularMAE,
        num_x: torch.Tensor,
        cat_x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = model(num_x, cat_x, mask)
        losses = []
        ce_loss = nn.CrossEntropyLoss()

        for spec in model.preprocessor.specs:
            column_mask = mask[:, spec.position]
            if not bool(column_mask.any()):
                continue

            if spec.kind == "numerical":
                idx = model.preprocessor.num_position_to_index[spec.position]
                target = num_x[:, idx]
                prediction = outputs["num_predictions"][spec.position]
                losses.append(((prediction[column_mask] - target[column_mask]) ** 2).mean())
            else:
                idx = model.preprocessor.cat_position_to_index[spec.position]
                target = cat_x[:, idx]
                logits = outputs["cat_predictions"][spec.position]
                losses.append(ce_loss(logits[column_mask], target[column_mask]))

        if not losses:
            return torch.tensor(0.0, device=num_x.device)
        return torch.stack(losses).mean()

    def train(self, device: str | torch.device) -> dict[str, object]:
        set_seed(self.config.seed)
        bundle = self.load_bundle()
        preprocessor = self.build_preprocessor()
        model = self.build_model().to(device)

        encoded = preprocessor.encode_dataframe(bundle.train_df)
        dataset = TensorDataset(
            torch.from_numpy(encoded.num).float(),
            torch.from_numpy(encoded.cat).long(),
        )
        batch_size = min(self.config.batch_size, len(bundle.train_df))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        history: list[float] = []
        model.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            seen = 0
            for num_batch, cat_batch in loader:
                num_batch = num_batch.to(device)
                cat_batch = cat_batch.to(device)
                mask = sample_column_mask(
                    batch_size=num_batch.size(0),
                    n_columns=len(preprocessor.specs),
                    mask_ratio=self.config.mask_ratio,
                    device=device,
                )

                loss = self._compute_masked_loss(model, num_batch, cat_batch, mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size_now = num_batch.size(0)
                epoch_loss += float(loss.item()) * batch_size_now
                seen += batch_size_now

            mean_loss = epoch_loss / max(seen, 1)
            history.append(mean_loss)
            print(f"[COSMIC][train] epoch={epoch + 1}/{self.config.epochs} loss={mean_loss:.6f}")

        latent = self.encode_latent(bundle.train_df, device)
        plan = compute_generation_plan(bundle, preprocessor, latent, self.config)
        conditional_models = fit_conditional_models(
            bundle=bundle,
            preprocessor=preprocessor,
            latent=latent,
            config=self.config,
            ordered_columns=plan.ordered_columns,
        )

        self.save_artifacts(model, preprocessor, plan, conditional_models, history)
        return {
            "history": history,
            "ordered_columns": plan.ordered_columns,
        }

    @torch.no_grad()
    def encode_latent(self, frame: pd.DataFrame, device: str | torch.device) -> np.ndarray:
        model = self.build_model().to(device)
        model.eval()
        preprocessor = self.build_preprocessor()
        num_tensor, cat_tensor = preprocessor.to_tensors(frame, device)

        latents = []
        batch_size = min(self.config.batch_size, len(frame))
        for start in range(0, len(frame), batch_size):
            end = start + batch_size
            latent, _ = model.encode(num_tensor[start:end], cat_tensor[start:end])
            latents.append(latent.detach().cpu().numpy())
        return np.concatenate(latents, axis=0)

    def save_artifacts(
        self,
        model: TabularMAE,
        preprocessor: TabularPreprocessor,
        plan: GenerationPlan,
        conditional_models: dict[str, ConditionalModel],
        history: list[float],
    ) -> None:
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.config.run_dir / "mae.pt")

        with (self.config.run_dir / "preprocessor.pkl").open("wb") as fp:
            pickle.dump(preprocessor, fp)

        with (self.config.run_dir / "conditional_models.pkl").open("wb") as fp:
            pickle.dump(conditional_models, fp)

        plan.to_json(self.config.run_dir / "generation_plan.json")
        (self.config.run_dir / "train_history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        config_payload = asdict(self.config)
        config_payload["run_dir"] = str(config_payload["run_dir"])
        (self.config.run_dir / "config.json").write_text(
            json.dumps(config_payload, indent=2),
            encoding="utf-8",
        )

    def load_artifacts(self, device: str | torch.device) -> tuple[TabularPreprocessor, TabularMAE, GenerationPlan, dict[str, ConditionalModel]]:
        config_path = self.config.run_dir / "config.json"
        artifact_config = {}
        if config_path.exists():
            artifact_config = json.loads(config_path.read_text(encoding="utf-8"))

        with (self.config.run_dir / "preprocessor.pkl").open("rb") as fp:
            self.preprocessor = pickle.load(fp)

        self.bundle = load_dataset_bundle(self.config.dataname)
        self.model = TabularMAE(
            preprocessor=self.preprocessor,
            hidden_dim=int(artifact_config.get("hidden_dim", self.config.hidden_dim)),
            depth=int(artifact_config.get("depth", self.config.depth)),
            num_heads=int(artifact_config.get("num_heads", self.config.num_heads)),
            dropout=float(artifact_config.get("dropout", self.config.dropout)),
        ).to(device)
        self.model.load_state_dict(torch.load(self.config.run_dir / "mae.pt", map_location=device))
        self.model.eval()
        self.artifact_context_dim = int(artifact_config.get("context_dim", self.config.context_dim))

        with (self.config.run_dir / "conditional_models.pkl").open("rb") as fp:
            conditional_models = pickle.load(fp)

        plan = GenerationPlan.from_json(self.config.run_dir / "generation_plan.json")
        return self.preprocessor, self.model, plan, conditional_models

    def _initial_synthetic_frame(self, bundle: DatasetBundle, n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
        data = {}
        for column in bundle.all_columns:
            values = bundle.train_df[column].to_numpy()
            data[column] = rng.choice(values, size=n_samples, replace=True)
        return pd.DataFrame(data)[bundle.all_columns].copy()

    def sample(self, device: str | torch.device, num_samples: int | None, save_path: str | Path) -> pd.DataFrame:
        set_seed(self.config.seed)
        bundle = self.load_bundle()
        preprocessor, model, plan, conditional_models = self.load_artifacts(device)

        n_samples = num_samples or bundle.train_size
        rng = np.random.default_rng(self.config.seed)
        synthetic = self._initial_synthetic_frame(bundle, n_samples, rng)

        for round_idx in range(self.config.mice_rounds):
            print(f"[COSMIC][sample] round={round_idx + 1}/{self.config.mice_rounds}")
            for column in plan.ordered_columns:
                latent = self.encode_latent(synthetic, device)
                context = latent[:, : min(self.artifact_context_dim, latent.shape[1])]
                synthetic[column] = conditional_models[column].sample(synthetic, context, rng)

        synthetic = synthetic[bundle.all_columns].copy()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(save_path, index=False)
        print(f"[COSMIC][sample] saved synthetic data to {save_path}")
        return synthetic
