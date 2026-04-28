import torch
from pathlib import Path

from baselines.common import get_artifact_dir, load_dataset_bundle, save_pickle


def _load_ctgan_class():
    try:
        from ctgan import CTGAN
    except ImportError as exc:
        raise ImportError(
            "CTGAN is not installed. Install the `ctgan` package to run this baseline."
        ) from exc

    return CTGAN


def _parse_dims(raw_dims: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw_dims.split(",") if part.strip())


def main(args):
    CTGAN = _load_ctgan_class()
    bundle = load_dataset_bundle(args.dataname)
    enable_gpu = torch.cuda.is_available() and args.gpu >= 0

    model_kwargs = dict(
        embedding_dim=args.embedding_dim,
        generator_dim=_parse_dims(args.generator_dim),
        discriminator_dim=_parse_dims(args.discriminator_dim),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=True,
    )
    try:
        model = CTGAN(**model_kwargs, enable_gpu=enable_gpu)
    except TypeError:
        model = CTGAN(**model_kwargs, cuda=enable_gpu)

    model.fit(bundle.frame, discrete_columns=bundle.discrete_columns)

    artifact_dir = get_artifact_dir(args.dataname, "ctgan")
    artifact_path = Path(artifact_dir) / "model.pkl"
    save_pickle(model, artifact_path)
    print(f"Saved CTGAN checkpoint to {artifact_path}")
