from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class COSMICConfig:
    dataname: str
    run_dir: Path
    mask_ratio: float = 0.3
    hidden_dim: int = 256
    depth: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    shap_estimator: str = "tree"
    order_strategy: str = "shap_desc"
    mice_rounds: int = 1
    context_dim: int = 32
    num_trees: int = 100
    shap_num_samples: int = 512
    seed: int = 42
