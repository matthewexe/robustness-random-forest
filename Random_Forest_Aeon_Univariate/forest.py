import json
import os

from tree import DecisionTree
from typing import List, Dict, Any, Union

RESULTS_DIR = "results"


def _results_path(filename: str) -> str:
    """Return a path inside the results directory (creating it if missing)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, os.path.basename(filename))


class Forest:
    def __init__(self, trees_data: List[Dict[str, Any]]):
        if not isinstance(trees_data, list):
            raise TypeError("trees_data must be a list")

        if len(trees_data) == 0:
            raise ValueError("trees_data cannot be empty")

        # Extract tree_ids and validate
        tree_ids = []
        for i, tree_dict in enumerate(trees_data):
            if not isinstance(tree_dict, dict):
                raise TypeError(f"Tree at index {i} must be a dictionary")

            if "tree_id" not in tree_dict:
                raise ValueError(f"Tree at index {i} must contain 'tree_id'")

            tree_id = tree_dict["tree_id"]
            if not isinstance(tree_id, int):
                raise TypeError(f"tree_id at index {i} must be an integer")

            tree_ids.append(tree_id)

        # Validate tree_ids start from 0 and have no holes
        tree_ids.sort()
        expected_ids = list(range(len(tree_ids)))

        if tree_ids != expected_ids:
            raise ValueError(f"tree_ids must start from 0 and have no holes. Expected {expected_ids}, got {tree_ids}")

        # Check for duplicates (should not happen after sorting check, but being extra safe)
        if len(set(tree_ids)) != len(tree_ids):
            raise ValueError("tree_ids must be unique")

        # Create DecisionTree objects and store them
        self.trees = []
        for tree_dict in trees_data:
            decision_tree = DecisionTree(tree_dict)
            self.trees.append(decision_tree)

        # Sort trees by tree_id to ensure consistent ordering
        self.trees.sort(key=lambda tree: tree.root["tree_id"])

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        return self.trees[index]

    def get_tree_by_id(self, tree_id: int):
        """Get a tree by its tree_id"""
        if not isinstance(tree_id, int):
            raise TypeError("tree_id must be an integer")

        if tree_id < 0 or tree_id >= len(self.trees):
            raise IndexError(f"tree_id {tree_id} out of range [0, {len(self.trees)-1}]")

        return self.trees[tree_id]

    def extract_feature_thresholds(self):
        """
        Extract all threshold values for each feature across all trees in the forest.
        Merges thresholds from all trees, maintaining unique values and sorting.

        Returns:
            Dictionary mapping feature names to strictly monotonically increasing arrays
            starting with -inf and ending with +inf, containing all unique thresholds
            for that feature across all trees.
        """
        all_feature_thresholds = {}

        # Collect thresholds from all trees
        for tree in self.trees:
            tree_thresholds = tree.extract_feature_thresholds()

            for feature, thresholds in tree_thresholds.items():
                if feature not in all_feature_thresholds:
                    all_feature_thresholds[feature] = set()

                # Add all thresholds except the -inf and +inf endpoints
                # (we'll add these back at the end)
                for threshold in thresholds:
                    if threshold not in [float('-inf'), float('inf')]:
                        all_feature_thresholds[feature].add(threshold)

        # Convert sets to sorted arrays with -inf and +inf endpoints
        result = {}
        for feature, thresholds in all_feature_thresholds.items():
            # Sort unique thresholds and add endpoints
            sorted_thresholds = sorted(list(thresholds))
            monotonic_array = [float('-inf')] + sorted_thresholds + [float('inf')]
            result[feature] = monotonic_array

        return result

    def predict(self, sample_dict):
        """
        Predict the label for a given sample using majority voting across all trees.

        Args:
            sample_dict: Dictionary with feature_name: float_value pairs

        Returns:
            str: The predicted label from majority voting across all trees

        Raises:
            ValueError: If required features are missing from sample_dict
            TypeError: If sample_dict is not a dictionary
        """
        if not isinstance(sample_dict, dict):
            raise TypeError("sample_dict must be a dictionary")

        if len(self.trees) == 0:
            raise ValueError("Forest has no trees")

        # Get predictions from all trees
        predictions = []
        for tree in self.trees:
            prediction = tree.predict(sample_dict)
            predictions.append(prediction)

        # Count votes for each label
        vote_counts = {}
        for prediction in predictions:
            if prediction in vote_counts:
                vote_counts[prediction] += 1
            else:
                vote_counts[prediction] = 1

        # Find the label with the most votes
        max_votes = max(vote_counts.values())

        # Get all labels with maximum votes (in case of ties)
        winning_labels = [label for label, count in vote_counts.items() if count == max_votes]

        # If there's a tie, return the first one (could be made more sophisticated)
        # In practice, ties are rare with enough trees
        winning_label = winning_labels[0]

        return winning_label


def store_forest(forest: Forest, filename:str) -> bool:
    if not isinstance(forest, Forest):
        raise TypeError("forest must be a Forest instance")

    if not isinstance(filename, str):
        raise TypeError("key must be a string")

    try:
        # Extract the raw tree data from each DecisionTree in the forest
        trees_data = []
        for tree in forest.trees:
            trees_data.append(tree.root)

        json_data = json.dumps(trees_data)
        filepath = _results_path(filename)
        with open(filepath, 'w') as f:
            f.write(json_data)

        print(f"Successfully stored forest with {len(trees_data)} trees.")
        return True
    except Exception as e:
        print(f"Error storing forest: {e}")
        return False


def retrieve_forest(filename="RF.json") -> Union[Forest, None]:
    try:
        filepath = _results_path(filename)
        with open(filepath, 'r') as f:
            json_data = f.read()

        if json_data is None:
            print(f"No forest found for '{filename}'")
            return None

        trees_data = json.loads(json_data)

        # Validate retrieved data
        if not isinstance(trees_data, list):
            raise ValueError("Retrieved data is not a list")

        # Create and return Forest object (this will validate the structure)
        forest = Forest(trees_data)
        print(f"Successfully retrieved forest with {len(forest)} trees")
        return forest

    except Exception as e:
        print(f"Error retrieving forest: {e}")
        return None
