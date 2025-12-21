from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .psf import Formula
    from .random_forest import RandomForest, DecisionTree, Node, LeafNode, InternalNode

_RF_Type = RandomForest
_DT_Type = DecisionTree
_DT_Node_Type = Node
_DT_Leaf_Type = LeafNode
_DT_Internal_Type = InternalNode
_Formula_Type = Formula
