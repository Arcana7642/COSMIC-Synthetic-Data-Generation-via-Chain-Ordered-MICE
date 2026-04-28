from pathlib import Path

from baselines.common import (
    ensure_parent,
    get_artifact_dir,
    get_default_sample_path,
    load_dataset_bundle,
    load_pickle,
)


def main(args):
    bundle = load_dataset_bundle(args.dataname)
    artifact_path = Path(get_artifact_dir(args.dataname, "tvae")) / "model.pkl"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"TVAE checkpoint not found at {artifact_path}. "
            f"Run: python main.py --dataname {args.dataname} --method tvae --mode train --epochs {args.epochs}"
        )

    model = load_pickle(artifact_path)
    sample_count = args.num_samples or len(bundle.frame)
    sampled = model.sample(sample_count)

    save_path = ensure_parent(args.save_path or get_default_sample_path(args.dataname, "tvae"))
    sampled.to_csv(save_path, index=False)
    print(f"Saved TVAE samples to {save_path}")
