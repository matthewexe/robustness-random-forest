from robustness.schemas.random_forest import (
    DecisionTreeSchema,
    InternalNodeSchema,
    LeafNodeSchema,
    RandomForestSchema,
)
from robustness.domain.random_forest import (
    RandomForest,
    DecisionTree,
    InternalNode,
    LeafNode,
    Node,
)


def to_model(rf_schema: RandomForestSchema) -> RandomForest:
    return RandomForest([__dt_to_model(dt) for dt in rf_schema])


def __dt_to_model(dt: DecisionTreeSchema) -> DecisionTree:
    root = __dt_internal_to_model(dt)
    return DecisionTree(tree_id=dt.tree_id, root=root)


def __dt_internal_to_model(node: InternalNodeSchema) -> InternalNode:
    low_child = None
    if isinstance(node.low_child, LeafNodeSchema):
        low_child = __dt_leaf_to_model(node.low_child)
    else:
        low_child = __dt_internal_to_model(node.low_child)

    high_child = None
    if isinstance(node.high_child, LeafNodeSchema):
        high_child = __dt_leaf_to_model(node.high_child)
    else:
        high_child = __dt_internal_to_model(node.high_child)

    return InternalNode(low_child, high_child, feature=node.feature, value=node.value)


def __dt_leaf_to_model(node: LeafNodeSchema, parent: Node | None = None) -> LeafNode:
    return LeafNode(label=node.label, leaf_id=node.leaf_id)
