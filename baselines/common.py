import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class DatasetBundle:
    name: str
    info: dict[str, Any]
    frame: pd.DataFrame
    data_dir: Path

    @property
    def discrete_columns(self) -> list[str]:
        idx_to_name = self.idx_to_name
        return [idx_to_name[idx] for idx in self.info.get("cat_col_idx", [])]

    @property
    def idx_to_name(self) -> dict[int, str]:
        mapping = self.info.get("idx_name_mapping")
        if mapping:
            return {int(k): v for k, v in mapping.items()}

        names = self.info.get("column_names") or list(self.frame.columns)
        return {idx: name for idx, name in enumerate(names)}


def resolve_info_path(dataname: str) -> Path:
    data_dir = ROOT_DIR / "data" / dataname
    direct = data_dir / "info.json"
    if direct.exists():
        return direct

    info_dir = ROOT_DIR / "data" / "Info" / f"{dataname}.json"
    if info_dir.exists():
        return info_dir

    raise FileNotFoundError(
        f"Could not find dataset metadata for '{dataname}'. "
        f"Expected either {direct} or {info_dir}."
    )


def load_dataset_bundle(dataname: str) -> DatasetBundle:
    info_path = resolve_info_path(dataname)
    with info_path.open("r", encoding="utf-8") as fp:
        info = json.load(fp)

    data_dir = ROOT_DIR / "data" / dataname
    csv_path = info.get("data_path")
    if csv_path:
        csv_path = (ROOT_DIR / csv_path).resolve()
    else:
        csv_path = data_dir / f"{dataname}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find raw dataset CSV at {csv_path}")

    frame = pd.read_csv(csv_path)
    return DatasetBundle(name=dataname, info=info, frame=frame, data_dir=data_dir)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_artifact_dir(dataname: str, method: str) -> Path:
    path = ROOT_DIR / "ckpt" / dataname / method
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_default_sample_path(dataname: str, method: str) -> Path:
    path = ROOT_DIR / "synthetic" / dataname / f"{method}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as fp:
        pickle.dump(obj, fp)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as fp:
        return pickle.load(fp)
