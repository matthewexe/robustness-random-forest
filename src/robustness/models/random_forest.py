import abc

from pydantic import BaseModel


class Node(BaseModel, abc.ABC):
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


class LeafNode(Node):
    def __init__(self, leaf_id: int, label: str):
        """
        Initialize a LeafNode.

        Args:
            leaf_id (int): Unique identifier for the leaf node.
            label (str): The class label assigned to this leaf node.

        Attributes:
            leaf_id (int): Unique identifier for the leaf node.
            label (str): The class label assigned to this leaf node.
        """

    leaf_id: int
    label: str


class DecisionTree(Node):
    """
    A decision tree node representing the root of a complete tree structure.

    Attributes:
        tree_id (int): Unique identifier for the decision tree.
    """

    tree_id: int


RandomForest = list[DecisionTree]
