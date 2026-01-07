from typing import TypeAlias

from robustness.domain.psf.model import PSF
from robustness.domain.random_forest import (
    RandomForest,
    DecisionTree,
    Node,
    LeafNode,
    InternalNode,
)

# Random Forest
_RF_Type: TypeAlias = RandomForest
_DT_Type: TypeAlias = DecisionTree
_DT_Node_Type: TypeAlias = Node
_DT_Leaf_Type: TypeAlias = LeafNode
_DT_Internal_Type: TypeAlias = InternalNode

# PSF
_PSF_Type: TypeAlias = PSF
