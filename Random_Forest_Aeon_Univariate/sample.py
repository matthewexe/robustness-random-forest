"""
Sample converter for sklearn data to dictionary format.
Handles conversion between sklearn array format and our dictionary representation.
"""
import json
import os

import numpy as np
from typing import List, Dict, Union, Optional

RESULTS_DIR = "results"


def _results_path(filename: str) -> str:
    """Return a path inside the results directory (creating it if missing)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, os.path.basename(filename))


def sklearn_sample_to_dict(sample: Union[np.ndarray, List],
                          feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Convert sklearn sample (numpy array or list) to dictionary format.

    Args:
        sample: Input sample as numpy array or list of feature values
        feature_names: List of feature names. If None, uses "feature_0", "feature_1", etc.

    Returns:
        Dictionary with feature_name: float_value pairs

    Raises:
        ValueError: If sample is empty or feature_names length doesn't match sample length
    """
    # Convert to numpy array if it's a list
    if isinstance(sample, list):
        sample = np.array(sample)
    elif not isinstance(sample, np.ndarray):
        raise ValueError(f"sample must be numpy array or list, got {type(sample)}")

    # Handle 1D and 2D arrays (flatten if needed)
    if sample.ndim == 0:
        raise ValueError("sample cannot be a scalar")
    elif sample.ndim > 2:
        raise ValueError(f"sample must be 1D or 2D array, got {sample.ndim}D")
    elif sample.ndim == 2:
        if sample.shape[0] != 1:
            raise ValueError(f"2D sample must have shape (1, n_features), got {sample.shape}")
        sample = sample.flatten()

    n_features = len(sample)
    if n_features == 0:
        raise ValueError("sample cannot be empty")

    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    elif len(feature_names) != n_features:
        raise ValueError(f"feature_names length ({len(feature_names)}) must match "
                        f"sample length ({n_features})")

    # Convert to dictionary with float values
    result = {}
    for i, (name, value) in enumerate(zip(feature_names, sample)):
        try:
            result[name] = float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert feature '{name}' at index {i} to float: {e}")

    return result


def dict_to_sklearn_sample(sample_dict: Dict[str, float],
                          feature_names: Optional[List[str]] = None) -> np.ndarray:
    """
    Convert dictionary format back to sklearn sample (numpy array).

    Args:
        sample_dict: Dictionary with feature_name: float_value pairs
        feature_names: Ordered list of feature names. If None, uses sorted keys from dict

    Returns:
        Numpy array with feature values in the specified order

    Raises:
        ValueError: If feature_names contains keys not in sample_dict
    """
    if not isinstance(sample_dict, dict):
        raise ValueError(f"sample_dict must be a dictionary, got {type(sample_dict)}")

    if len(sample_dict) == 0:
        raise ValueError("sample_dict cannot be empty")

    # Use sorted keys if no feature names provided
    if feature_names is None:
        feature_names = sorted(sample_dict.keys())

    # Check that all feature names exist in the dictionary
    missing_features = set(feature_names) - set(sample_dict.keys())
    if missing_features:
        raise ValueError(f"Features not found in sample_dict: {sorted(missing_features)}")

    # Extract values in the specified order
    values = []
    for name in feature_names:
        try:
            value = float(sample_dict[name])
            values.append(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert feature '{name}' to float: {e}")

    return np.array(values)


def store_sample(sample_dict: Dict[str, float], filename:str) -> bool:
    try:
        # Validate input
        if not isinstance(sample_dict, dict):
            raise ValueError(f"sample_dict must be a dictionary, got {type(sample_dict)}")

        if len(sample_dict) == 0:
            raise ValueError("sample_dict cannot be empty")

        # Validate that all values can be converted to float
        validated_dict = {}
        for feature, value in sample_dict.items():
            try:
                validated_dict[feature] = float(value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot convert feature '{feature}' to float: {e}")

        filepath = _results_path(filename)
        with open(filepath, 'w') as f:
            f.write(json.dumps(validated_dict))

        return True

    except Exception as e:
        print(f"Error storing sample: {e}")
        return False


def retrieve_sample(filename:str) -> Optional[Dict[str, float]]:
    try:
        filepath = _results_path(filename)
        with open(filepath, 'r') as f:
            data_str = f.read()

        if data_str is None:
            print(f"No sample found in '{filename}'")
            return None

        # Parse JSON
        sample_dict = json.loads(data_str)

        # Validate structure
        if not isinstance(sample_dict, dict):
            raise ValueError("Invalid sample data structure")

        print(f"Successfully retrieved sample with {len(sample_dict)} features from '{filename}'")
        return sample_dict

    except json.JSONDecodeError as e:
        print(f"Error parsing: '{filename}': {e}")
        return None
    except Exception as e:
        print(f"Error retrieving sample from: '{filename}': {e}")
        return None
