from typing import List, Dict, Any
from sktree_to_tree import sklearn_tree_to_dict
from forest import Forest


def sklearn_forest_to_forest(sklearn_forest, feature_names: List[str] = None, class_names: List[str] = None) -> Forest:
    """
    Convert a sklearn ensemble (RandomForestClassifier, etc.) to a Forest object.
    Trees are assigned progressive IDs starting from 0.

    Args:
        sklearn_forest: Trained sklearn ensemble classifier with estimators_ attribute
        feature_names: List of feature names (optional, will use indices if not provided)
        class_names: List of class labels (optional, will use indices if not provided)

    Returns:
        Forest object with converted trees
    """
    if not hasattr(sklearn_forest, 'estimators_'):
        raise ValueError("sklearn_forest must be a fitted ensemble classifier with estimators_ attribute")

    # Get class names from the forest if not provided
    if class_names is None:
        if hasattr(sklearn_forest, 'classes_'):
            # Forest has classes_ attribute, individual trees don't
            class_names = [str(cls) for cls in sklearn_forest.classes_]
        else:
            # Fallback: try to infer from first tree
            first_tree = sklearn_forest.estimators_[0]
            if hasattr(first_tree, 'tree_'):
                tree_structure = first_tree.tree_
                n_classes = tree_structure.n_classes[0] if tree_structure.n_outputs == 1 else max(tree_structure.n_classes)
                class_names = [str(i) for i in range(n_classes)]
            else:
                raise ValueError("Cannot determine class names from sklearn_forest")

    trees_data = []
    for tree_id, estimator in enumerate(sklearn_forest.estimators_):
        # Pass the forest-level class names and parent forest to each tree conversion
        tree_dict = sklearn_tree_to_dict(estimator, tree_id, feature_names, class_names, sklearn_forest)
        trees_data.append(tree_dict)

    return Forest(trees_data)


def visualize_forests_side_by_side(sklearn_forest, our_forest: Forest,
                                 feature_names: List[str] = None, max_trees: int = 3) -> str:
    """
    Create side-by-side visualization of sklearn forest and converted Forest.
    Shows up to max_trees trees from each forest.

    Args:
        sklearn_forest: Trained sklearn ensemble classifier
        our_forest: Converted Forest object
        feature_names: List of feature names (optional)
        max_trees: Maximum number of trees to visualize (default 3)

    Returns:
        DOT format string with both forests side by side
    """
    if not hasattr(sklearn_forest, 'estimators_'):
        raise ValueError("sklearn_forest must be a fitted ensemble classifier with estimators_ attribute")

    import numpy as np

    # Limit number of trees to visualize
    n_trees_to_show = min(max_trees, len(sklearn_forest.estimators_), len(our_forest))

    if feature_names is None:
        n_features = sklearn_forest.estimators_[0].tree_.n_features
        feature_names = [f"feature_{i}" for i in range(n_features)]

    dot_lines = ['digraph ForestComparison {']
    dot_lines.append('    rankdir=TB;')
    dot_lines.append('    node [shape=box, style=filled, fontname="Arial"];')
    dot_lines.append('    ')

    # Left side: sklearn forest
    dot_lines.append('    subgraph cluster_sklearn {')
    dot_lines.append('        label="Original sklearn Forest";')
    dot_lines.append('        style=filled;')
    dot_lines.append('        color=lightgrey;')
    dot_lines.append('        ')

    for tree_idx in range(n_trees_to_show):
        estimator = sklearn_forest.estimators_[tree_idx]
        tree_structure = estimator.tree_

        dot_lines.append(f'        subgraph cluster_sk_tree_{tree_idx} {{')
        dot_lines.append(f'            label="Tree {tree_idx}";')
        dot_lines.append('            style=filled;')
        dot_lines.append('            color=white;')

        def _add_sklearn_nodes(node_id: int, tree_idx: int):
            node_name = f"sk_t{tree_idx}_n{node_id}"

            if tree_structure.children_left[node_id] == tree_structure.children_right[node_id]:
                # Leaf node
                values = tree_structure.value[node_id][0]
                predicted_class = np.argmax(values)
                samples = int(tree_structure.n_node_samples[node_id])
                label = f"Class: {predicted_class}\\nSamples: {samples}"
                dot_lines.append(f'            {node_name} [label="{label}", fillcolor="lightgreen"];')
            else:
                # Internal node
                feature_idx = tree_structure.feature[node_id]
                threshold = tree_structure.threshold[node_id]
                samples = int(tree_structure.n_node_samples[node_id])
                label = f"{feature_names[feature_idx]} <= {threshold:.3f}\\nSamples: {samples}"
                dot_lines.append(f'            {node_name} [label="{label}", fillcolor="lightblue"];')

                # Add children
                left_child = tree_structure.children_left[node_id]
                right_child = tree_structure.children_right[node_id]

                if left_child != -1:
                    left_name = f"sk_t{tree_idx}_n{left_child}"
                    dot_lines.append(f'            {node_name} -> {left_name} [label="≤", color="blue"];')
                    _add_sklearn_nodes(left_child, tree_idx)

                if right_child != -1:
                    right_name = f"sk_t{tree_idx}_n{right_child}"
                    dot_lines.append(f'            {node_name} -> {right_name} [label=">", color="red"];')
                    _add_sklearn_nodes(right_child, tree_idx)

        _add_sklearn_nodes(0, tree_idx)
        dot_lines.append('        }')
        dot_lines.append('        ')

    dot_lines.append('    }')
    dot_lines.append('    ')

    # Right side: converted forest
    dot_lines.append('    subgraph cluster_converted {')
    dot_lines.append('        label="Converted Forest";')
    dot_lines.append('        style=filled;')
    dot_lines.append('        color=lightcyan;')
    dot_lines.append('        ')

    for tree_idx in range(n_trees_to_show):
        tree = our_forest[tree_idx]
        tree_dict = tree.root

        dot_lines.append(f'        subgraph cluster_cv_tree_{tree_idx} {{')
        dot_lines.append(f'            label="Tree {tree_idx}";')
        dot_lines.append('            style=filled;')
        dot_lines.append('            color=white;')

        node_counter = [0]

        def _add_converted_nodes(node_data: Dict[str, Any], tree_idx: int, is_root: bool = False) -> str:
            node_name = f"cv_t{tree_idx}_n{node_counter[0]}"
            node_counter[0] += 1

            if "leaf_id" in node_data and "label" in node_data:
                # Leaf node
                label = f"Leaf ID: {node_data['leaf_id']}\\nClass: {node_data['label']}"
                if is_root:
                    label = f"Tree ID: {node_data.get('tree_id', tree_dict['tree_id'])}\\n{label}"
                dot_lines.append(f'            {node_name} [label="{label}", fillcolor="lightgreen"];')
            elif "feature" in node_data and "value" in node_data:
                # Internal node
                label = f"{node_data['feature']} <= {node_data['value']:.3f}"
                if is_root:
                    label = f"Tree ID: {tree_dict['tree_id']}\\n{label}"
                dot_lines.append(f'            {node_name} [label="{label}", fillcolor="lightblue"];')

                # Add children
                if "low_child" in node_data:
                    left_name = _add_converted_nodes(node_data["low_child"], tree_idx, False)
                    dot_lines.append(f'            {node_name} -> {left_name} [label="≤", color="blue"];')

                if "high_child" in node_data:
                    right_name = _add_converted_nodes(node_data["high_child"], tree_idx, False)
                    dot_lines.append(f'            {node_name} -> {right_name} [label=">", color="red"];')

            return node_name

        # Process root node (excluding tree_id)
        root_data = {k: v for k, v in tree_dict.items() if k != "tree_id"}
        _add_converted_nodes(root_data, tree_idx, True)

        dot_lines.append('        }')
        dot_lines.append('        ')

    dot_lines.append('    }')
    dot_lines.append('}')

    return '\n'.join(dot_lines)