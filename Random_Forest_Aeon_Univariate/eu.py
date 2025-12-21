import json
import math
import os
from typing import Dict, List, Union

RESULTS_DIR = "results"


def _results_path(filename: str) -> str:
	"""Return a path inside the results directory (creating it if missing)."""
	os.makedirs(RESULTS_DIR, exist_ok=True)
	return os.path.join(RESULTS_DIR, os.path.basename(filename))

def validate_monotonic_array(arr: List[float], key_name: str) -> bool:
	"""Validate that array starts with -inf, ends with +inf, and is monotonically increasing"""
	if not isinstance(arr, list):
		raise TypeError(f"Array for key '{key_name}' must be a list")

	if len(arr) < 2:
		raise ValueError(f"Array for key '{key_name}' must have at least 2 elements")

	if not math.isinf(arr[0]) or arr[0] >= 0:
		raise ValueError(f"Array for key '{key_name}' must start with -infinity")

	if not math.isinf(arr[-1]) or arr[-1] <= 0:
		raise ValueError(f"Array for key '{key_name}' must end with +infinity")

	# Check monotonically increasing
	for i in range(1, len(arr)):
		if arr[i] <= arr[i-1]:
			raise ValueError(f"Array for key '{key_name}' must be monotonically increasing at index {i}")

	return True

def store_monotonic_dict(data: Dict[str, List[float]], filename:str) -> bool:
	"""Store a dictionary of key:array pairs where arrays are monotonically increasing from -inf to +inf"""
	if not isinstance(data, dict):
		raise TypeError("Data must be a dictionary")

	# Validate all arrays
	for dict_key, array in data.items():
		if not isinstance(dict_key, str):
			raise TypeError(f"Dictionary key '{dict_key}' must be a string")
		validate_monotonic_array(array, dict_key)

	try:
		json_data = json.dumps(data)
		filepath = _results_path(filename)
		with open(filepath, 'w') as f:
			f.write(json_data)

		print(f"Successfully stored dictionary with {len(data)} keys in file '{filename}'")

		return True
	except Exception as e:
		print(f"Error storing data: {e}")
		return False

def retrieve_monotonic_dict(filename:str) -> Union[Dict[str, List[float]], None]:
	try:
		filepath = _results_path(filename)
		with open(filepath, 'r') as f:
			json_data = f.read()

		if json_data is None:
			print(f"No data found in file '{filename}'")
			return None

		data = json.loads(json_data)

		# Validate retrieved data
		if not isinstance(data, dict):
			raise ValueError("Retrieved data is not a dictionary")

		for dict_key, array in data.items():
			validate_monotonic_array(array, dict_key)

		print(f"Successfully retrieved dictionary with {len(data)} keys from file '{filename}'")
		return data

	except Exception as e:
		print(f"Error retrieving data: {e}")
		return None
