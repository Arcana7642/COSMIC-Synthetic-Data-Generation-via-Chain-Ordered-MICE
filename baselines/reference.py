import json
import shutil
import subprocess
import sys
from pathlib import Path

from baselines.common import ROOT_DIR, ensure_parent, resolve_info_path


SAMPLE_METHODS = {"smote", "stasy", "codi", "tabddpm", "tabsyn"}
TRAIN_METHODS = SAMPLE_METHODS | {"vae"}


def resolve_reference_root(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (ROOT_DIR / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Reference repository not found: {path}")

    return path


def sync_dataset_to_reference(dataname: str, reference_root: Path) -> None:
    source_dir = ROOT_DIR / "data" / dataname
    target_dir = reference_root / "data" / dataname
    target_dir.mkdir(parents=True, exist_ok=True)

    if source_dir.exists():
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)

    info_source = resolve_info_path(dataname)
    info_target_dir = reference_root / "data" / "Info"
    info_target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(info_source, info_target_dir / f"{dataname}.json")

    with info_source.open("r", encoding="utf-8") as fp:
        info = json.load(fp)

    if not (target_dir / "info.json").exists():
        shutil.copy2(info_source, target_dir / "info.json")

    csv_path = info.get("data_path")
    if csv_path:
        csv_path = (ROOT_DIR / csv_path).resolve()
        if csv_path.exists():
            target_csv = target_dir / csv_path.name
            if not target_csv.exists():
                shutil.copy2(csv_path, target_csv)


def run_reference_method(args, method: str, mode: str) -> None:
    if method not in TRAIN_METHODS:
        raise ValueError(f"Unsupported reference method: {method}")
    if mode == "sample" and method not in SAMPLE_METHODS:
        raise ValueError(f"Sampling is not supported for reference method: {method}")

    reference_root = resolve_reference_root(args.reference_root)
    sync_dataset_to_reference(args.dataname, reference_root)

    command = [
        sys.executable,
        "main.py",
        "--dataname",
        args.dataname,
        "--method",
        method,
        "--mode",
        mode,
        "--gpu",
        str(args.gpu),
    ]

    if mode == "sample":
        save_path = ensure_parent(args.save_path).resolve()
        command.extend(["--save_path", str(save_path)])
        if method == "tabddpm":
            command.extend(["--steps", str(args.steps)])
            if args.ddim:
                command.append("--ddim")
    elif method == "smote":
        command.extend(["--cat_encoding", args.cat_encoding])

    print(f"Delegating to reference TabSyn repo at {reference_root}")
    print(" ".join(command))

    subprocess.run(command, cwd=reference_root, check=True)
