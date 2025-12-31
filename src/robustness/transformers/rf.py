from robustness.schemas.random_forest import (
    DecisionTreeSchema,
    InternalNodeSchema,
    LeafNodeSchema,
    RandomForestSchema, SampleSchema,
)
from robustness.domain.random_forest import (
    RandomForest,
    DecisionTree,
    InternalNode,
    LeafNode,
    Node, Sample,
)


def rf_schema_to_model(rf_schema: RandomForestSchema) -> RandomForest:
    return RandomForest([__dt_to_model(dt) for dt in rf_schema])


def __dt_to_model(dt: DecisionTreeSchema) -> DecisionTree:
    root = __dt_node_to_model(dt)
    return DecisionTree(tree_id=dt.tree_id, root=root)


def __dt_node_to_model(node: InternalNodeSchema | LeafNodeSchema) -> InternalNode | LeafNode:
    if isinstance(node, LeafNodeSchema):
        return LeafNode(label=node.label, leaf_id=node.leaf_id)

    low_child = __dt_node_to_model(node.low_child)
    high_child = __dt_node_to_model(node.high_child)

    return InternalNode(low_child, high_child, feature=node.feature, value=node.value)

def sample_schema_to_model(sample_schema: SampleSchema):
    return Sample(**sample_schema)