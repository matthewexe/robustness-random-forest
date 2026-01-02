from robustness.domain.types import (
    _RF_Type,
    _DT_Node_Type,
    _DT_Internal_Type,
    _DT_Leaf_Type,
)
from robustness.domain.logging import get_logger

logger = get_logger(__name__)


# def rf_to_formula_tree(rf: _RF_Type) -> _Formula_Type:
#     conditions = [c for tree in rf for c in dt_to_formula_tree(tree.root)]
#     if len(conditions) < 1:
#         return Constant(True)
#
#     def rec(_conditions) -> _Formula_Type:
#         if len(_conditions) == 1:
#             return _conditions[0]
#         return Or(_conditions[0], rec(_conditions[1:]))
#
#     return rec(conditions)
#
#
# def dt_to_formula_tree(root: _DT_Node_Type) -> Sequence[_Formula_Type]:
#     if isinstance(root, _DT_Internal_Type):
#         low_child_path_conditions = dt_to_formula_tree(root.low_child)
#         low_condition = Not(Variable(root.feature))
#         low_paths = []
#         if len(low_child_path_conditions) == 0:
#             low_paths = [low_condition]
#         else:
#             for condition in low_child_path_conditions:
#                 low_paths.append(And(low_condition, condition))
#
#         high_child_path_conditions = dt_to_formula_tree(root.high_child)
#         high_condition = Variable(root.feature)
#         high_paths = []
#         if len(high_child_path_conditions) == 0:
#             high_paths = [high_condition]
#         else:
#             for condition in high_child_path_conditions:
#                 high_paths.append(And(high_condition, condition))
#
#         return low_paths + high_paths
#     if isinstance(root, _DT_Leaf_Type):
#         return []
#
#     return []


def rf_to_formula_str(rf: _RF_Type) -> str:
    logger.info(f"Converting random forest with {len(rf)} trees to formula string")
    groups = [c for tree in rf for c in dt_to_formula_str(tree.root)]
    and_expr = [" and ".join(reversed(group)) for group in groups]
    formula = "(" + (") or (".join(and_expr)) + ")"
    logger.debug(f"Generated formula string with {len(groups)} groups and length {len(formula)}")
    return formula


def dt_to_formula_str(root: _DT_Node_Type) -> list[list[str]]:
    logger.debug(f"Converting decision tree node to formula string")
    if isinstance(root, _DT_Internal_Type):
        low_child_path_conditions = dt_to_formula_str(root.low_child)
        low_condition = f"(not {root.feature})"
        low_paths: list[list[str]] = []
        if len(low_child_path_conditions) == 0:
            low_paths = [[low_condition]]
        else:
            for condition in low_child_path_conditions:
                condition_copy = condition[:]
                condition_copy.append(low_condition)
                low_paths.append(condition_copy)

        high_child_path_conditions = dt_to_formula_str(root.high_child)
        high_condition = root.feature
        high_paths: list[list[str]] = []
        if len(high_child_path_conditions) == 0:
            high_paths = [[high_condition]]
        else:
            for condition in high_child_path_conditions:
                condition_copy = condition[:]
                condition_copy.append(high_condition)
                high_paths.append(condition_copy)

        return low_paths + high_paths
    if isinstance(root, _DT_Leaf_Type):
        return [[f"c{root.label}"]]

    return [[]]
