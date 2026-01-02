from os import PathLike
from pathlib import Path

from robustness.domain.logging import get_logger

logger = get_logger(__name__)


def get_file_content(path: str | PathLike[str]) -> str:
    logger.debug(f"Reading file content from: {path}")
    with open(path, "r") as f:
        content = f.read()
    logger.debug(f"Successfully read {len(content)} bytes from {path}")
    return content


class RandomForestService:
    __model_name: str
    __base_results_path: Path

    def __init__(self, model_name: str, base_results_path: str) -> None:
        logger.info(f"Initializing RandomForestService for model: {model_name}")
        logger.debug(f"Base results path: {base_results_path}")
        self.__model_name = model_name
        self.__base_results_path = Path(base_results_path)

    def get_random_forest(self) -> str:
        path = self.__base_results_path / self.get_rf_file_name()
        logger.info(f"Loading random forest from: {path}")
        content = get_file_content(path.resolve())
        logger.debug(f"Random forest loaded successfully from {path}")
        return content

    def get_sample(self, sample_id: str) -> str:
        path = self.__base_results_path / self.get_sample_file_name(sample_id)
        logger.info(f"Loading sample {sample_id} from: {path}")
        content = get_file_content(path.resolve())
        logger.debug(f"Sample {sample_id} loaded successfully")
        return content

    def get_rf_file_name(self) -> str:
        filename = f"{self.__model_name}_random_forest.json"
        logger.debug(f"Generated random forest filename: {filename}")
        return filename

    def get_sample_file_name(self, sample_id:str) -> str:
        filename = f"sample_{self.__model_name}_{sample_id}.json"
        logger.debug(f"Generated sample filename: {filename}")
        return filename