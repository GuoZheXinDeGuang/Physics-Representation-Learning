from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path, overrides: Sequence[str] = ()) -> DictConfig:
    """Load a YAML config and apply Hydra-style dotlist overrides."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg


def set_the_well_env(cfg: DictConfig) -> str:
    """Set THE_WELL_DATA_DIR from config before constructing physics_jepa datasets."""
    configured_dir = Path(str(cfg.paths.get("the_well_data_dir", "/home_shared/grail_enoch/the_well")))
    dataset_name = str(cfg.dataset.get("name", "active_matter"))

    if (configured_dir / dataset_name / "data").is_dir():
        data_dir = configured_dir
    elif (configured_dir / "datasets" / dataset_name / "data").is_dir():
        data_dir = configured_dir / "datasets"
    else:
        data_dir = configured_dir

    os.environ["THE_WELL_DATA_DIR"] = str(data_dir)
    return os.environ["THE_WELL_DATA_DIR"]


def save_resolved_config(cfg: DictConfig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_path)
