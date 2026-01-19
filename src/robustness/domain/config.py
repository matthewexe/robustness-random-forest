from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from robustness.domain.utils.singleton import Singleton

root_path = Path().parent.parent.parent.parent


@dataclass
class Config(Singleton):
    diagram_size: int = 200
    prefix_var: str = "t_"
    dataset_name: str = "Meat"
    rf_path: str | PathLike[str] = root_path / "Random_Forest_Aeon_Univariate/results"
    debug_mode: bool = False
    log_graphs: bool = False
