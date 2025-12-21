from typing import Dict, Any, List
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def sklearn_tree_to_dict(sklearn_tree: DecisionTreeClassifier, tree_id: int, feature_names: List[str] = None, class_names: List[str] = None, parent_forest=None) -> Dict[str, Any]:
    """
    Convert a sklearn DecisionTreeClassifier to a dictionary format compatible with DecisionTree class.

    Args:
        sklearn_tree: Trained sklearn DecisionTreeClassifier
        tree_id: Integer ID for the tree
        feature_names: List of feature names (optional, will use indices if not provided)
        class_names: List of class labels (optional, will use indices if not provided)
        parent_forest: The parent forest object to get correct class mapping (optional)

    Returns:
        Dictionary representation of the tree compatible with DecisionTree class
    """
    if not hasattr(sklearn_tree, 'tree_'):
        raise ValueError("sklearn_tree must be a fitted DecisionTreeClassifier")

    tree_structure = sklearn_tree.tree_

    # Use feature names or default to feature indices
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(tree_structure.n_features)]

    # Get class names - prefer parent forest, then explicit class_names, then tree's classes_
    if class_names is None:
        if parent_forest is not None and hasattr(parent_forest, 'classes_'):
            # Use parent forest's classes (this is the correct mapping)
            class_names = [str(cls) for cls in parent_forest.classes_]
        elif hasattr(sklearn_tree, 'classes_'):
            # If classes_ exists on tree, use it (individual DecisionTree case)
            class_names = [str(cls) for cls in sklearn_tree.classes_]
        else:
            # Fallback to indices
            n_classes = tree_structure.n_classes[0] if tree_structure.n_outputs == 1 else max(tree_structure.n_classes)
            class_names = [str(i) for i in range(n_classes)]

    def _convert_node(node_id: int, leaf_counter: List[int]) -> Dict[str, Any]:
        """
        Recursively convert sklearn tree nodes to dictionary format.

        Args:
            node_id: Current node ID in sklearn tree
            leaf_counter: Mutable list with single element to track leaf IDs

        Returns:
            Dictionary representation of the node
        """
        # Check if this is a leaf node
        if tree_structure.children_left[node_id] == tree_structure.children_right[node_id]:
            # Leaf node
            # Get the most common class in this leaf
            values = tree_structure.value[node_id][0]
            predicted_class_idx = np.argmax(values)

            leaf_node = {
                "leaf_id": leaf_counter[0],
                "label": class_names[predicted_class_idx]
            }
            leaf_counter[0] += 1
            return leaf_node

        else:
            # Internal node
            feature_idx = tree_structure.feature[node_id]
            threshold = tree_structure.threshold[node_id]

            left_child_id = tree_structure.children_left[node_id]
            right_child_id = tree_structure.children_right[node_id]

            internal_node = {
                "feature": feature_names[feature_idx],
                "value": float(threshold),
                "low_child": _convert_node(left_child_id, leaf_counter),
                "high_child": _convert_node(right_child_id, leaf_counter)
            }

            return internal_node

    # Start conversion from root (node 0)
    leaf_counter = [0]  # Mutable counter for leaf IDs
    root_node = _convert_node(0, leaf_counter)

    # Add tree_id to the root
    tree_dict = {"tree_id": tree_id}
    tree_dict.update(root_node)

    return tree_dict


def sklearn_forest_to_dicts(sklearn_forest, feature_names: List[str] = None, class_names: List[str] = None) -> List[Dict[str, Any]]:
    """
    Convert a sklearn ensemble (RandomForestClassifier, etc.) to a list of dictionaries
    compatible with the Forest class.

    Args:
        sklearn_forest: Trained sklearn ensemble classifier with estimators_ attribute
        feature_names: List of feature names (optional, will use indices if not provided)
        class_names: List of class labels (optional, will use indices if not provided)

    Returns:
        List of dictionary representations compatible with Forest class
    """
    if not hasattr(sklearn_forest, 'estimators_'):
        raise ValueError("sklearn_forest must be a fitted ensemble classifier with estimators_ attribute")

    trees_data = []
    for tree_id, estimator in enumerate(sklearn_forest.estimators_):
        tree_dict = sklearn_tree_to_dict(estimator, tree_id, feature_names, class_names, sklearn_forest)
        trees_data.append(tree_dict)

    return trees_data


def tree_to_dot(tree_dict: Dict[str, Any], graph_name: str = "Tree") -> str:
    """
    Convert our DecisionTree dictionary format to DOT notation for Graphviz.

    Args:
        tree_dict: Dictionary representation of tree (compatible with DecisionTree class)
        graph_name: Name for the graph

    Returns:
        DOT format string for Graphviz visualization
    """
    dot_lines = [f'digraph {graph_name} {{']
    dot_lines.append('    node [shape=box, style=filled, fontname="Arial"];')

    node_counter = [0]

    def _add_node(node_data: Dict[str, Any], parent_id: int = None, edge_label: str = None) -> int:
        current_id = node_counter[0]
        node_counter[0] += 1

        if "leaf_id" in node_data and "label" in node_data:
            # Leaf node
            label = f"Leaf {node_data['leaf_id']}\\nClass: {node_data['label']}"
            dot_lines.append(f'    {current_id} [label="{label}", fillcolor="lightgreen"];')
        elif "feature" in node_data and "value" in node_data:
            # Internal node
            label = f"{node_data['feature']} <= {node_data['value']:.3f}"
            dot_lines.append(f'    {current_id} [label="{label}", fillcolor="lightblue"];')

            # Add children
            if "low_child" in node_data:
                left_id = _add_node(node_data["low_child"], current_id, "≤")
                dot_lines.append(f'    {current_id} -> {left_id} [label="≤", color="blue"];')

            if "high_child" in node_data:
                right_id = _add_node(node_data["high_child"], current_id, ">")
                dot_lines.append(f'    {current_id} -> {right_id} [label=">", color="red"];')

        return current_id

    # Process root node (excluding tree_id)
    root_data = {k: v for k, v in tree_dict.items() if k != "tree_id"}
    _add_node(root_data)

    dot_lines.append('}')
    return '\n'.join(dot_lines)


def sklearn_tree_to_dot(sklearn_tree: DecisionTreeClassifier, feature_names: List[str] = None,
                       graph_name: str = "SklearnTree") -> str:
    """
    Convert sklearn DecisionTreeClassifier to DOT notation for Graphviz.

    Args:
        sklearn_tree: Trained sklearn DecisionTreeClassifier
        feature_names: List of feature names (optional)
        graph_name: Name for the graph

    Returns:
        DOT format string for Graphviz visualization
    """
    if not hasattr(sklearn_tree, 'tree_'):
        raise ValueError("sklearn_tree must be a fitted DecisionTreeClassifier")

    tree_structure = sklearn_tree.tree_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(tree_structure.n_features)]

    dot_lines = [f'digraph {graph_name} {{']
    dot_lines.append('    node [shape=box, style=filled, fontname="Arial"];')

    def _add_sklearn_node(node_id: int):
        # Check if leaf
        if tree_structure.children_left[node_id] == tree_structure.children_right[node_id]:
            # Leaf node
            values = tree_structure.value[node_id][0]
            predicted_class = np.argmax(values)
            samples = int(tree_structure.n_node_samples[node_id])

            label = f"Class: {predicted_class}\\nSamples: {samples}"
            dot_lines.append(f'    {node_id} [label="{label}", fillcolor="lightgreen"];')
        else:
            # Internal node
            feature_idx = tree_structure.feature[node_id]
            threshold = tree_structure.threshold[node_id]
            samples = int(tree_structure.n_node_samples[node_id])

            label = f"{feature_names[feature_idx]} <= {threshold:.3f}\\nSamples: {samples}"
            dot_lines.append(f'    {node_id} [label="{label}", fillcolor="lightblue"];')

            # Add edges to children
            left_child = tree_structure.children_left[node_id]
            right_child = tree_structure.children_right[node_id]

            if left_child != -1:
                dot_lines.append(f'    {node_id} -> {left_child} [label="≤", color="blue"];')
                _add_sklearn_node(left_child)

            if right_child != -1:
                dot_lines.append(f'    {node_id} -> {right_child} [label=">", color="red"];')
                _add_sklearn_node(right_child)

    _add_sklearn_node(0)  # Start from root

    dot_lines.append('}')
    return '\n'.join(dot_lines)


def visualize_trees_side_by_side(sklearn_tree: DecisionTreeClassifier, tree_dict: Dict[str, Any],
                                feature_names: List[str] = None) -> str:
    """
    Create side-by-side visualization of sklearn tree and converted tree.

    Args:
        sklearn_tree: Trained sklearn DecisionTreeClassifier
        tree_dict: Converted tree dictionary
        feature_names: List of feature names (optional)

    Returns:
        DOT format string with both trees side by side
    """
    if not hasattr(sklearn_tree, 'tree_'):
        raise ValueError("sklearn_tree must be a fitted DecisionTreeClassifier")

    tree_structure = sklearn_tree.tree_
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(tree_structure.n_features)]

    dot_lines = ['digraph Comparison {']
    dot_lines.append('    rankdir=TB;')
    dot_lines.append('    node [shape=box, style=filled, fontname="Arial"];')
    dot_lines.append('    ')

    # Left side: sklearn tree
    dot_lines.append('    subgraph cluster_0 {')
    dot_lines.append('        label="Original sklearn Tree";')
    dot_lines.append('        style=filled;')
    dot_lines.append('        color=lightgrey;')
    dot_lines.append('        rank=same;')

    def _add_sklearn_nodes(node_id: int, prefix: str = "sk"):
        node_name = f"{prefix}_{node_id}"

        if tree_structure.children_left[node_id] == tree_structure.children_right[node_id]:
            # Leaf node
            values = tree_structure.value[node_id][0]
            predicted_class = np.argmax(values)
            samples = int(tree_structure.n_node_samples[node_id])
            label = f"Class: {predicted_class}\\nSamples: {samples}"
            dot_lines.append(f'        {node_name} [label="{label}", fillcolor="lightgreen"];')
        else:
            # Internal node
            feature_idx = tree_structure.feature[node_id]
            threshold = tree_structure.threshold[node_id]
            samples = int(tree_structure.n_node_samples[node_id])
            label = f"{feature_names[feature_idx]} <= {threshold:.3f}\\nSamples: {samples}"
            dot_lines.append(f'        {node_name} [label="{label}", fillcolor="lightblue"];')

            # Add children
            left_child = tree_structure.children_left[node_id]
            right_child = tree_structure.children_right[node_id]

            if left_child != -1:
                left_name = f"{prefix}_{left_child}"
                dot_lines.append(f'        {node_name} -> {left_name} [label="≤", color="blue"];')
                _add_sklearn_nodes(left_child, prefix)

            if right_child != -1:
                right_name = f"{prefix}_{right_child}"
                dot_lines.append(f'        {node_name} -> {right_name} [label=">", color="red"];')
                _add_sklearn_nodes(right_child, prefix)

    _add_sklearn_nodes(0)
    dot_lines.append('    }')
    dot_lines.append('    ')

    # Right side: converted tree
    dot_lines.append('    subgraph cluster_1 {')
    dot_lines.append('        label="Converted Tree";')
    dot_lines.append('        style=filled;')
    dot_lines.append('        color=lightcyan;')
    dot_lines.append('        rank=same;')

    node_counter = [0]

    def _add_converted_nodes(node_data: Dict[str, Any], prefix: str = "cv", is_root: bool = False) -> str:
        node_name = f"{prefix}_{node_counter[0]}"
        node_counter[0] += 1

        if "leaf_id" in node_data and "label" in node_data:
            # Leaf node - always show leaf_id since it's consistently assigned
            label = f"Leaf ID: {node_data['leaf_id']}\\nClass: {node_data['label']}"
            if is_root:
                label = f"Tree ID: {tree_dict['tree_id']}\\n{label}"
            dot_lines.append(f'        {node_name} [label="{label}", fillcolor="lightgreen"];')
        elif "feature" in node_data and "value" in node_data:
            # Internal node
            label = f"{node_data['feature']} <= {node_data['value']:.3f}"
            if is_root:
                label = f"Tree ID: {tree_dict['tree_id']}\\n{label}"
            dot_lines.append(f'        {node_name} [label="{label}", fillcolor="lightblue"];')

            # Add children
            if "low_child" in node_data:
                left_name = _add_converted_nodes(node_data["low_child"], prefix, False)
                dot_lines.append(f'        {node_name} -> {left_name} [label="≤", color="blue"];')

            if "high_child" in node_data:
                right_name = _add_converted_nodes(node_data["high_child"], prefix, False)
                dot_lines.append(f'        {node_name} -> {right_name} [label=">", color="red"];')

        return node_name

    # Process root node (excluding tree_id)
    root_data = {k: v for k, v in tree_dict.items() if k != "tree_id"}
    _add_converted_nodes(root_data, "cv", True)

    dot_lines.append('    }')
    dot_lines.append('}')

    return '\n'.join(dot_lines)