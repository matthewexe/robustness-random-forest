from __future__ import annotations
import abc


class Node(abc.ABC):
    """
    A Node model representing a node in a tree structure.

    This class serves as a base model for tree nodes used in tree-based
    machine learning models such as decision trees and random forests.

    Attributes:
        None (currently empty, inherits from BaseModel)

    Note:
        This is a placeholder class that should be extended with specific
        node attributes such as feature index, threshold, left/right children,
        and prediction values depending on the tree implementation.
    """

    parent: Node | None

    def __init__(self, parent: Node | None = None) -> None:
        self.parent = parent

    def is_root(self) -> bool:
        return self.parent is None

    def set_parent(self, parent: Node | None) -> None:
        self.parent = parent


class InternalNode(Node):
    """
    Internal node of a decision tree in a random forest.

    Attributes:
        low_child (Node): Child node containing samples with feature values less than the split value.
        high_child (Node): Child node containing samples with feature values greater than or equal to the split value.
        feature (str): Name of the feature used for splitting at this node.
        value (float): Threshold value used to split samples at this node.
    """

    low_child: Node
    high_child: Node
    feature: str
    value: float

    def __init__(
        self,
        low_child: Node,
        high_child: Node,
        feature: str,
        value: float,
        parent: Node | None = None,
    ) -> None:
        super().__init__(parent)
        self.low_child = low_child
        self.high_child = high_child
        self.feature = feature
        self.value = value

        # Set parent
        self.low_child.set_parent(self)
        self.high_child.set_parent(self)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, InternalNode):
            return False

        return (
            value.low_child == self.low_child
            and value.high_child == self.high_child
            and value.feature == self.feature
            and value.value == self.value
        )


class LeafNode(Node):
    """
    A leaf node in a decision tree representing a final classification.
    This class represents a terminal node in a decision tree that contains
    a classification label. Leaf nodes are reached when the tree traversal
    is complete and no further splits are possible.
    Attributes:
        leaf_id (int): Unique identifier for this leaf node.
        label (str): The classification label assigned to this leaf node.
    """

    leaf_id: int
    label: str

    def __init__(
        self,
        leaf_id: int,
        label: str,
        parent: Node | None = None,
    ) -> None:
        super().__init__(parent)
        self.leaf_id = leaf_id
        self.label = label

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, LeafNode)
            and value.label == self.label
            and value.leaf_id == self.leaf_id
        )


class DecisionTree:
    """
    A decision tree node representing the root of a complete tree structure.
    Assuming that a decision tree contains at least one internal node.

    Attributes:
        tree_id (int): Unique identifier for the decision tree.
    """

    tree_id: int
    root: Node

    def __init__(self, tree_id: int, root: Node) -> None:
        self.tree_id = tree_id

        # Force parent to be None
        root.set_parent(None)
        self.root = root

    def __eq__(self, value: object) -> bool:
        return isinstance(value, DecisionTree) and value.tree_id == self.tree_id


class RandomForest(list[DecisionTree]):
    pass
