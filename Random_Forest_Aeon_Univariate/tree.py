class DecisionTree:
    def __init__(self, tree_dict):
        if not isinstance(tree_dict, dict):
            raise TypeError("tree_dict must be a dictionary")

        if "tree_id" not in tree_dict:
            raise ValueError("tree_dict must contain 'tree_id'")

        if not isinstance(tree_dict["tree_id"], int):
            raise TypeError("tree_id must be an integer")

        root_node = {k: v for k, v in tree_dict.items() if k != "tree_id"}

        self._validate_node(root_node)

        self.root = tree_dict

        # Automatically assign leaf_ids based on depth-first traversal
        self.assign_leaf_ids()

    def _validate_node(self, node):
        if not isinstance(node, dict):
            raise TypeError("node must be a dictionary")

        if "leaf_id" in node and "label" in node:
            if len(node) != 2:
                raise ValueError("leaf node must contain exactly 'leaf_id' and 'label'")
            if not isinstance(node["leaf_id"], int):
                raise TypeError("leaf_id must be an integer")
            if not isinstance(node["label"], str):
                raise TypeError("label must be a string")

        elif "feature" in node and "value" in node and "low_child" in node and "high_child" in node:
            if len(node) != 4:
                raise ValueError("internal node must contain exactly 'feature', 'value', 'low_child', and 'high_child'")
            if not isinstance(node["feature"], str):
                raise TypeError("feature must be a string")
            if not isinstance(node["value"], (int, float)):
                raise TypeError("value must be a number")

            self._validate_node(node["low_child"])
            self._validate_node(node["high_child"])

        else:
            raise ValueError("node must be either a leaf node (leaf_id, label) or internal node (feature, value, low_child, high_child)")

    def assign_leaf_ids(self):
        """
        Reassign leaf_ids based on depth-first traversal with low_child visited first.
        This ensures consistent leaf numbering regardless of the original leaf_id values.
        """
        leaf_counter = [0]

        def _reassign_leaf_ids(node):
            if "leaf_id" in node and "label" in node:
                # Leaf node - assign new leaf_id
                node["leaf_id"] = leaf_counter[0]
                leaf_counter[0] += 1
            elif "feature" in node and "value" in node and "low_child" in node and "high_child" in node:
                # Internal node - traverse children with low_child first
                _reassign_leaf_ids(node["low_child"])
                _reassign_leaf_ids(node["high_child"])

        # Start traversal from root (excluding tree_id)
        root_node = {k: v for k, v in self.root.items() if k != "tree_id"}
        _reassign_leaf_ids(root_node)

        # Store the number of leaves
        self.n_leaves = leaf_counter[0]

    def extract_feature_thresholds(self):
        """
        Extract all threshold values for each feature in the tree.

        Returns:
            Dictionary mapping feature names to strictly monotonically increasing arrays
            starting with -inf and ending with +inf, containing all thresholds for that feature.
        """
        feature_thresholds = {}

        def _collect_thresholds(node):
            if "feature" in node and "value" in node and "low_child" in node and "high_child" in node:
                # Internal node - collect threshold
                feature = node["feature"]
                threshold = node["value"]

                if feature not in feature_thresholds:
                    feature_thresholds[feature] = set()

                feature_thresholds[feature].add(threshold)

                # Recursively collect from children
                _collect_thresholds(node["low_child"])
                _collect_thresholds(node["high_child"])

        # Start collection from root (excluding tree_id)
        root_node = {k: v for k, v in self.root.items() if k != "tree_id"}
        _collect_thresholds(root_node)

        # Convert sets to sorted arrays with -inf and +inf endpoints
        result = {}
        for feature, thresholds in feature_thresholds.items():
            # Sort thresholds and add endpoints
            sorted_thresholds = sorted(list(thresholds))
            monotonic_array = [float('-inf')] + sorted_thresholds + [float('inf')]
            result[feature] = monotonic_array

        return result

    def predict(self, sample_dict):
        """
        Predict the label for a given sample.

        Args:
            sample_dict: Dictionary with feature_name: float_value pairs

        Returns:
            str: The predicted label from the leaf node

        Raises:
            ValueError: If required features are missing from sample_dict
            TypeError: If sample_dict is not a dictionary
        """
        if not isinstance(sample_dict, dict):
            raise TypeError("sample_dict must be a dictionary")

        def _traverse(node):
            # Check if this is a leaf node
            if "leaf_id" in node and "label" in node:
                return node["label"]

            # Internal node - make decision
            if "feature" in node and "value" in node and "low_child" in node and "high_child" in node:
                feature = node["feature"]
                threshold = node["value"]

                # Check if feature exists in sample
                if feature not in sample_dict:
                    raise ValueError(f"Feature '{feature}' not found in sample_dict")

                sample_value = sample_dict[feature]

                # Apply test: sample[feature] <= threshold
                if sample_value <= threshold:
                    return _traverse(node["low_child"])
                else:
                    return _traverse(node["high_child"])

            raise ValueError("Invalid node structure encountered during prediction")

        # Start traversal from root (excluding tree_id)
        root_node = {k: v for k, v in self.root.items() if k != "tree_id"}
        return _traverse(root_node)

    def extract_icf(self, sample_dict):
        """
        Extract Interval Condition Features (ICF) for a given sample.

        Returns a dictionary where each feature maps to an open interval (inf, sup)
        representing the path conditions the sample traversed through the tree.

        Args:
            sample_dict: Dictionary with feature_name: float_value pairs

        Returns:
            dict: Feature name -> (inf, sup) interval where inf < sup
                  Intervals include ±infinity as bounds

        Raises:
            ValueError: If required features are missing from sample_dict
            TypeError: If sample_dict is not a dictionary
        """
        if not isinstance(sample_dict, dict):
            raise TypeError("sample_dict must be a dictionary")

        # Initialize intervals for all features with (-inf, +inf)
        icf = {}

        def _traverse_and_collect(node, current_intervals):
            # Copy current intervals to avoid modifying parent's state
            intervals = current_intervals.copy()

            # Check if this is a leaf node
            if "leaf_id" in node and "label" in node:
                return intervals

            # Internal node - make decision and update intervals
            if "feature" in node and "value" in node and "low_child" in node and "high_child" in node:
                feature = node["feature"]
                threshold = node["value"]

                # Check if feature exists in sample
                if feature not in sample_dict:
                    raise ValueError(f"Feature '{feature}' not found in sample_dict")

                sample_value = sample_dict[feature]

                # Initialize interval for this feature if not seen before
                if feature not in intervals:
                    intervals[feature] = (float('-inf'), float('inf'))

                # Apply test: sample[feature] <= threshold
                if sample_value <= threshold:
                    # Go to low_child, update upper bound
                    inf, sup = intervals[feature]
                    intervals[feature] = (inf, min(sup, threshold))
                    return _traverse_and_collect(node["low_child"], intervals)
                else:
                    # Go to high_child, update lower bound
                    inf, sup = intervals[feature]
                    intervals[feature] = (max(inf, threshold), sup)
                    return _traverse_and_collect(node["high_child"], intervals)

            raise ValueError("Invalid node structure encountered during ICF extraction")

        # Start traversal from root (excluding tree_id)
        root_node = {k: v for k, v in self.root.items() if k != "tree_id"}
        icf = _traverse_and_collect(root_node, {})

        return icf

    def get_icf_profile(self, icf):
        """
        Compute the profile of an ICF in the tree using the traversal algorithm.

        The profile is the set of leaf IDs that can be reached within the ICF constraints.

        Args:
            icf: Dictionary mapping feature names to (inf, sup) intervals

        Returns:
            set: Set of leaf IDs that can be reached within the ICF
        """
        if not isinstance(icf, dict):
            raise TypeError("icf must be a dictionary")

        # Validate ICF intervals
        for feature, interval in icf.items():
            if not isinstance(interval, tuple) or len(interval) != 2:
                raise ValueError(f"ICF interval for feature '{feature}' must be a tuple of length 2")
            inf, sup = interval
            if inf >= sup:
                raise ValueError(f"ICF interval for feature '{feature}' must have inf < sup, got ({inf}, {sup})")

        def _traverse_node(node, current_icf):
            # If leaf, return singleton set of leaf id
            if "leaf_id" in node and "label" in node:
                return {node["leaf_id"]}

            # Internal node
            if "feature" in node and "value" in node and "low_child" in node and "high_child" in node:
                node_feature = node["feature"]
                threshold = node["value"]

                # Get ICF bounds for this feature
                if node_feature in current_icf:
                    icf_inf, icf_sup = current_icf[node_feature]
                else:
                    # Feature not constrained by ICF, use full range
                    icf_inf, icf_sup = float('-inf'), float('inf')

                # Case 1: ICF is entirely in low region (all icf <= threshold)
                if icf_sup <= threshold:
                    return _traverse_node(node["low_child"], current_icf)

                # Case 2: ICF is entirely in high region (all icf > threshold)
                elif icf_inf > threshold:
                    return _traverse_node(node["high_child"], current_icf)

                # Case 3: Threshold splits the ICF (icf_inf <= threshold < icf_sup)
                else:
                    # Split ICF into low and high parts
                    low_icf = current_icf.copy()
                    low_icf[node_feature] = (icf_inf, threshold)

                    high_icf = current_icf.copy()
                    high_icf[node_feature] = (threshold, icf_sup)

                    # Recursive calls on both children with split ICF
                    low_result = _traverse_node(node["low_child"], low_icf)
                    high_result = _traverse_node(node["high_child"], high_icf)

                    return low_result.union(high_result)

        # Start from root
        root_node = {k: v for k, v in self.root.items() if k != "tree_id"}
        return _traverse_node(root_node, icf)

    def get_max_leaf_id(self):
        """
        Get the maximum leaf ID in the tree.

        Returns:
            int: The maximum leaf ID
        """
        max_leaf_id = -1

        def _find_max_leaf_id(node):
            nonlocal max_leaf_id
            if "leaf_id" in node and "label" in node:
                max_leaf_id = max(max_leaf_id, node["leaf_id"])
            elif "feature" in node and "value" in node and "low_child" in node and "high_child" in node:
                _find_max_leaf_id(node["low_child"])
                _find_max_leaf_id(node["high_child"])

        root_node = {k: v for k, v in self.root.items() if k != "tree_id"}
        _find_max_leaf_id(root_node)
        return max_leaf_id

    def icf_profile_to_bitmap(self, icf):
        """
        Convert an ICF profile to a bitmap representation.

        After computing the ICF profile (set of leaf IDs), creates a bitmap of length
        n_leaves where position i has value 1 if leaf i is in the profile, 0 otherwise.

        Args:
            icf: Dictionary mapping feature names to (inf, sup) intervals

        Returns:
            list: Bitmap where bitmap[i] = 1 if leaf i is in the ICF profile, 0 otherwise
        """
        # Ensure n_leaves is available
        if not hasattr(self, 'n_leaves'):
            self.assign_leaf_ids()

        # Get the ICF profile (set of leaf IDs)
        profile = self.get_icf_profile(icf)

        # Create bitmap of length n_leaves
        bitmap = [0] * self.n_leaves

        # Set 1 for leaves in the profile
        for leaf_id in profile:
            bitmap[leaf_id] = 1

        return bitmap