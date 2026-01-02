from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from robustness.domain.utils.singleton import Singleton
from robustness.domain.logging import get_logger

logger = get_logger(__name__)

root_path = Path().parent.parent.parent.parent

@dataclass
class Config(Singleton):
    diagram_size: int = 200
    prefix_var: str = "t_"
    dataset_name: str = "Meat"
    rf_path: str | PathLike[str] = root_path / "Random_Forest_Aeon_Univariate/results"
    
    def __post_init__(self):
        # Note: This is only called once due to Singleton pattern
        logger.info(f"Config singleton initialized: dataset={self.dataset_name}, diagram_size={self.diagram_size}")
        logger.debug(f"Config details: prefix_var={self.prefix_var}, rf_path={self.rf_path}")
