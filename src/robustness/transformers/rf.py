import datetime

from robustness.domain.logging import get_logger
from robustness.domain.random_forest import (
    RandomForest,
    DecisionTree,
    InternalNode,
    LeafNode,
    Sample, Endpoints,
)
from robustness.schemas.random_forest import (
    DecisionTreeSchema,
    InternalNodeSchema,
    LeafNodeSchema,
    RandomForestSchema, SampleSchema, EndpointsSchema,
)

logger = get_logger(__name__)


def rf_schema_to_model(rf_schema: RandomForestSchema) -> RandomForest:
    logger.info(f"Converting RandomForestSchema to model with {len(rf_schema)} trees")
    result = RandomForest([__dt_to_model(dt) for dt in rf_schema])
    logger.debug(f"RandomForest model created with {len(result)} trees")
    return result


def __dt_to_model(dt: DecisionTreeSchema) -> DecisionTree:
    logger.debug(f"Converting DecisionTree schema for tree_id: {dt.tree_id}")
    root = __dt_node_to_model(dt)
    return DecisionTree(tree_id=dt.tree_id, root=root)


def __dt_node_to_model(node: InternalNodeSchema | LeafNodeSchema) -> InternalNode | LeafNode:
    if isinstance(node, LeafNodeSchema):
        logger.debug(f"Creating LeafNode: leaf_id={node.leaf_id}, label={node.label}")
        return LeafNode(label=node.label, leaf_id=node.leaf_id)

    logger.debug(f"Creating InternalNode: feature={node.feature}, value={node.value}")
    low_child = __dt_node_to_model(node.low_child)
    high_child = __dt_node_to_model(node.high_child)

    return InternalNode(low_child, high_child, feature=node.feature, value=node.value)


def sample_schema_to_model(group: int, _id: int, schema: SampleSchema) -> Sample:
    logger.debug(f"Converting SampleSchema to model with {len(schema.sample_dict)} entries")
    timestamp = datetime.datetime.fromisoformat(schema.timestamp)
    return Sample(group=group,
                  sample_id=_id,
                  features=schema.sample_dict.copy(),
                  predicted_label=schema.predicted_label,
                  actual_label=schema.actual_label,
                  dataset_name=schema.dataset_name,
                  timestamp=timestamp,
                  prediction_correct=schema.prediction_correct)


def endpoints_to_model(endpoints: EndpointsSchema) -> Endpoints:
    return {k: endpoints[k][1] for k in endpoints}
