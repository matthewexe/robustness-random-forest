from __future__ import annotations

import abc
from typing import TypeVar

from pydantic import BaseModel
from robustness.domain.logging import get_logger

logger = get_logger(__name__)


# Random Forest Schema
class NodeSchema(BaseModel, abc.ABC):
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


class InternalNodeSchema(NodeSchema):
    """
    Internal node of a decision tree in a random forest.

    Attributes:
        low_child (Node): Child node containing samples with feature values less than the split value.
        high_child (Node): Child node containing samples with feature values greater than or equal to the split value.
        feature (str): Name of the feature used for splitting at this node.
        value (float): Threshold value used to split samples at this node.
    """

    low_child: InternalNodeSchema | LeafNodeSchema
    high_child: InternalNodeSchema | LeafNodeSchema
    feature: str
    value: float


class LeafNodeSchema(NodeSchema):
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


class DecisionTreeSchema(InternalNodeSchema):
    """
    A decision tree node representing the root of a complete tree structure.
    Assuming that a decision tree contains at least one internal node.

    Attributes:
        tree_id (int): Unique identifier for the decision tree.
    """

    tree_id: int


RandomForestSchema = list[DecisionTreeSchema]

# Sample Schema
SampleSchema = dict[str, float]

T = TypeVar("T")


def from_json(json_content: str, cls: T) -> T:
    from pydantic import TypeAdapter
    
    logger.info(f"Deserializing JSON to {cls}")
    logger.debug(f"JSON content length: {len(json_content)}")

    adapter = TypeAdapter(cls)
    result = adapter.validate_json(json_content)
    logger.info(f"Successfully deserialized JSON to {cls}")
    return result
