import argparse
import importlib


SUPPORTED_METHODS = (
    "cosmic",
    "smote",
    "ctgan",
    "tvae",
    "stasy",
    "codi",
    "tabddpm",
    "tabsyn",
    "vae",
)


def execute_function(method: str, mode: str):
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}")

    if method == "cosmic":
        module_name = "COSMIC"
    elif method == "vae":
        module_name = "baselines.tabsyn.vae"
    elif method == "tabsyn":
        module_name = f"baselines.tabsyn.main" if mode == "train" else f"baselines.tabsyn.sample"
    elif method == "tabddpm":
        module_name = (
            "baselines.tabddpm.main_train"
            if mode == "train"
            else "baselines.tabddpm.main_sample"
        )
    else:
        module_name = f"baselines.{method}.main" if mode == "train" else f"baselines.{method}.sample"

    module = importlib.import_module(module_name)
    return getattr(module, "main")


def get_args():
    parser = argparse.ArgumentParser(description="COSMIC baseline runner")

    parser.add_argument("--dataname", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("train", "sample"),
        required=True,
        help="Whether to train or sample",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=SUPPORTED_METHODS,
        required=True,
        help="Baseline or model name",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for reference methods")
    parser.add_argument("--save_path", type=str, default=None, help="Output path for sampled CSV")

    parser.add_argument("--epochs", type=int, default=300, help="Epochs for CTGAN/TVAE")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for CTGAN/TVAE")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Latent dimension for CTGAN/TVAE")
    parser.add_argument(
        "--generator_dim",
        type=str,
        default="256,256",
        help="Comma-separated generator hidden dimensions for CTGAN",
    )
    parser.add_argument(
        "--discriminator_dim",
        type=str,
        default="256,256",
        help="Comma-separated discriminator hidden dimensions for CTGAN",
    )
    parser.add_argument(
        "--compress_dims",
        type=str,
        default="128,128",
        help="Comma-separated encoder hidden dimensions for TVAE",
    )
    parser.add_argument(
        "--decompress_dims",
        type=str,
        default="128,128",
        help="Comma-separated decoder hidden dimensions for TVAE",
    )
    parser.add_argument("--l2scale", type=float, default=1e-5, help="L2 regularization for TVAE")
    parser.add_argument("--loss_factor", type=int, default=2, help="TVAE reconstruction loss factor")
    parser.add_argument("--num_samples", type=int, default=None, help="Rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--ddim", action="store_true", default=False, help="Use DDIM for TabDDPM")
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps for diffusion methods")
    parser.add_argument("--cat_encoding", type=str, default="one-hot", help="Categorical encoding for SMOTE")
    parser.add_argument("--mask_ratio", type=float, default=0.3, help="Mask ratio for COSMIC masked autoencoder")
    parser.add_argument("--cosmic_hidden_dim", type=int, default=256, help="Hidden size for COSMIC modules")
    parser.add_argument("--cosmic_depth", type=int, default=4, help="Depth for COSMIC masked autoencoder")
    parser.add_argument("--cosmic_num_heads", type=int, default=8, help="Attention heads for COSMIC masked autoencoder")
    parser.add_argument("--cosmic_dropout", type=float, default=0.1, help="Dropout for COSMIC masked autoencoder")
    parser.add_argument("--cosmic_epochs", type=int, default=100, help="Training epochs for COSMIC")
    parser.add_argument("--cosmic_batch_size", type=int, default=512, help="Batch size for COSMIC")
    parser.add_argument("--cosmic_lr", type=float, default=1e-3, help="Learning rate for COSMIC")
    parser.add_argument("--cosmic_weight_decay", type=float, default=1e-5, help="Weight decay for COSMIC")
    parser.add_argument("--cosmic_context_dim", type=int, default=32, help="Latent context dimension for COSMIC conditional models")
    parser.add_argument("--shap_estimator", type=str, default="tree", help="SHAP estimator backend for COSMIC")
    parser.add_argument("--order_strategy", type=str, default="shap_desc", help="Column order strategy for COSMIC generation")
    parser.add_argument("--mice_rounds", type=int, default=1, help="Number of order-aware MICE refinement rounds")
    parser.add_argument("--mice_trees", type=int, default=100, help="Number of trees for COSMIC SHAP and MICE models")
    parser.add_argument("--shap_num_samples", type=int, default=512, help="Maximum number of rows used for SHAP estimation")

    return parser.parse_args()
