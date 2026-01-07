import glob
import os
import re
from os import PathLike
from pathlib import Path

from robustness.adapters import from_json
from robustness.domain.logging import get_logger
from robustness.schemas.random_forest import RandomForestSchema, SampleSchema, EndpointsSchema

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

    def get_rf_file_name(self) -> str:
        filename = f"{self.__model_name}_random_forest.json"
        logger.debug(f"Generated random forest filename: {filename}")
        return filename

    def random_forest(self) -> RandomForestSchema:
        path = self.__base_results_path / self.get_rf_file_name()
        logger.info(f"Loading random forest from: {path}")
        content = get_file_content(path.resolve())
        logger.debug(f"Random forest loaded successfully from {path}")
        return from_json(content, RandomForestSchema)

    def sample(self, group: int, sample_id: int) -> SampleSchema:
        path = self.__base_results_path / self.get_sample_file_name(group, sample_id)
        logger.info(f"Loading sample {group}_{sample_id} from: {path}")
        content = get_file_content(path.resolve())
        logger.debug(f"Sample {group}_{sample_id} loaded successfully")
        return SampleSchema.model_validate_json(content)

    def get_sample_file_name(self, group: int, sample_id: int) -> str:
        filename = f"sample_meta_{self.__model_name}_{group}_{sample_id}.json"
        logger.debug(f"Generated sample filename: {filename}")
        return filename

    def sample_ids(self) -> list[tuple[int, int]]:
        """Extract IDs using glob pattern matching"""
        files_pattern = self.get_sample_file_name("*", "*")
        pattern = self.__base_results_path / files_pattern
        filenames = glob.glob(str(pattern.resolve()))

        regex = rf'sample_meta_{self.__model_name}_(\d+)_(\d+)\.json'

        ids = []
        for filepath in filenames:
            filename = os.path.basename(filepath)
            match = re.match(regex, filename)
            if match:
                group = match.group(1)
                file_id = match.group(2)
                ids.append((int(group), int(file_id)))

        ids.sort()
        return ids

    def samples(self) -> dict[tuple[int, int], SampleSchema]:
        samples = {}
        for sample in self.sample_ids():
            group, sample_id = sample
            samples[sample] = self.sample(group, sample_id)

        return samples

    def get_endpoints_filename(self) -> str:
        return f"{self.__model_name}_endpoints_universe.json"

    def endpoints(self) -> EndpointsSchema:
        path = self.__base_results_path / self.get_endpoints_filename()
        content = get_file_content(path.resolve())
        return from_json(content, EndpointsSchema)
