import json

from robustness.models.random_forest import RandomForest


def get_rf_model_from_file(file_path: str) -> RandomForest:
    with open(file_path, "r") as f:
        _json = f.read()

    obj = json.loads(_json)

    from pydantic import TypeAdapter

    adapter = TypeAdapter(RandomForest)
    return adapter.validate_json(_json)
