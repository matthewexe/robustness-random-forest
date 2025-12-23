from pathlib import Path


class RandomForestService:
    __model_name: str
    __base_results_path: Path

    def __init__(self, model_name: str, base_results_path: str) -> None:
        self.__model_name = model_name
        self.__base_results_path = Path(base_results_path)

    def get_random_forest(self) -> str:
        path = self.__base_results_path / get_rf_file_name(self.__model_name)
        with open(path.resolve(), "r") as f:
            content = f.read()

        return content


def get_rf_file_name(model_name: str):
    return f"{model_name}_random_forest.json"
