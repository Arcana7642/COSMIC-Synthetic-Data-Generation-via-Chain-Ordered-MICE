import torch
from pathlib import Path

from baselines.common import get_artifact_dir, load_dataset_bundle, save_pickle


def _load_tvae_class():
    try:
        from ctgan import TVAE
    except ImportError as exc:
        raise ImportError(
            "TVAE is not installed. Install the `ctgan` package to run this baseline."
        ) from exc

    return TVAE


def _parse_dims(raw_dims: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw_dims.split(",") if part.strip())


def main(args):
    TVAE = _load_tvae_class()
    bundle = load_dataset_bundle(args.dataname)

    model = TVAE(
        embedding_dim=args.embedding_dim,
        compress_dims=_parse_dims(args.compress_dims),
        decompress_dims=_parse_dims(args.decompress_dims),
        batch_size=args.batch_size,
        epochs=args.epochs,
        l2scale=args.l2scale,
        loss_factor=args.loss_factor,
        cuda=torch.cuda.is_available() and args.gpu >= 0,
    )
    model.fit(bundle.frame, discrete_columns=bundle.discrete_columns)

    artifact_dir = get_artifact_dir(args.dataname, "tvae")
    artifact_path = Path(artifact_dir) / "model.pkl"
    save_pickle(model, artifact_path)
    print(f"Saved TVAE checkpoint to {artifact_path}")
