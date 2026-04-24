from __future__ import annotations

from typing import Any

import torch
from torch import nn

from cosmic.data import TabularPreprocessor


class TabularMAE(nn.Module):
    def __init__(
        self,
        preprocessor: TabularPreprocessor,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.hidden_dim = hidden_dim
        self.n_columns = len(preprocessor.specs)

        self.column_embeddings = nn.Parameter(torch.randn(self.n_columns, hidden_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        self.num_tokenizers = nn.ModuleDict()
        self.cat_tokenizers = nn.ModuleDict()
        self.num_heads = nn.ModuleDict()
        self.cat_heads = nn.ModuleDict()

        for spec in preprocessor.specs:
            key = str(spec.position)
            if spec.kind == "numerical":
                self.num_tokenizers[key] = nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.num_heads[key] = nn.Linear(hidden_dim, 1)
            else:
                n_classes = len(preprocessor.cat_classes[spec.name])
                self.cat_tokenizers[key] = nn.Embedding(n_classes, hidden_dim)
                self.cat_heads[key] = nn.Linear(hidden_dim, n_classes)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.final_norm = nn.LayerNorm(hidden_dim)

    def build_tokens(self, num_x: torch.Tensor, cat_x: torch.Tensor) -> torch.Tensor:
        tokens = []
        for spec in self.preprocessor.specs:
            key = str(spec.position)
            if spec.kind == "numerical":
                idx = self.preprocessor.num_position_to_index[spec.position]
                value = num_x[:, idx : idx + 1]
                token = self.num_tokenizers[key](value)
            else:
                idx = self.preprocessor.cat_position_to_index[spec.position]
                token = self.cat_tokenizers[key](cat_x[:, idx])
            token = token + self.column_embeddings[spec.position]
            tokens.append(token)
        return torch.stack(tokens, dim=1)

    def encode(self, num_x: torch.Tensor, cat_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.build_tokens(num_x, cat_x)
        hidden = self.final_norm(self.encoder(tokens))
        latent = hidden.mean(dim=1)
        return latent, hidden

    def forward(
        self,
        num_x: torch.Tensor,
        cat_x: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, Any]:
        tokens = self.build_tokens(num_x, cat_x)
        mask_tokens = self.mask_token.view(1, 1, -1) + self.column_embeddings.unsqueeze(0)
        masked_tokens = torch.where(mask.unsqueeze(-1), mask_tokens, tokens)
        hidden = self.final_norm(self.encoder(masked_tokens))
        latent = hidden.mean(dim=1)

        num_predictions: dict[int, torch.Tensor] = {}
        cat_predictions: dict[int, torch.Tensor] = {}

        for spec in self.preprocessor.specs:
            key = str(spec.position)
            state = hidden[:, spec.position]
            if spec.kind == "numerical":
                num_predictions[spec.position] = self.num_heads[key](state).squeeze(-1)
            else:
                cat_predictions[spec.position] = self.cat_heads[key](state)

        return {
            "latent": latent,
            "hidden": hidden,
            "num_predictions": num_predictions,
            "cat_predictions": cat_predictions,
        }


def sample_column_mask(
    batch_size: int,
    n_columns: int,
    mask_ratio: float,
    device: str | torch.device,
) -> torch.Tensor:
    mask = torch.rand(batch_size, n_columns, device=device) < mask_ratio
    missing = ~mask.any(dim=1)
    if missing.any():
        random_idx = torch.randint(0, n_columns, (int(missing.sum().item()),), device=device)
        mask[missing, random_idx] = True
    return mask
