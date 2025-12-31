from os import PathLike
from pathlib import Path


def get_file_content(path: str | PathLike[str]) -> str:
    with open(path, "r") as f:
        content = f.read()
    return content


class RandomForestService:
    __model_name: str
    __base_results_path: Path

    def __init__(self, model_name: str, base_results_path: str) -> None:
        self.__model_name = model_name
        self.__base_results_path = Path(base_results_path)

    def get_random_forest(self) -> str:
        path = self.__base_results_path / self.get_rf_file_name()
        return get_file_content(path.resolve())

    def get_sample(self, sample_id: str) -> str:
        path = self.__base_results_path / self.get_sample_file_name(sample_id)
        return get_file_content(path.resolve())

    def get_rf_file_name(self) -> str:
        return f"{self.__model_name}_random_forest.json"

    def get_sample_file_name(self, sample_id:str) -> str:
        return f"sample_{self.__model_name}_{sample_id}.json"