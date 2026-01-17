from __future__ import annotations

from collections import deque, Counter

import networkx as nx

import robustness.domain.psf.tableau.model as tb
from robustness.domain.bdd.manager import create_bdd_manager
from robustness.domain.bdd.operations import features_counting, Not, And
from robustness.domain.config import Config
from robustness.domain.logging import get_logger
from robustness.domain.psf.model import PSF, Kind, Builder, is_terminal, is_bdd
from robustness.domain.psf.tableau.model import TableauTree
from robustness.domain.tree.model import BinaryTree
from robustness.domain.tree.operations import map_nodes_of, remove_unconnected_nodes
from robustness.domain.utils.metrics import log_perf_counter

logger = get_logger(__name__)


def simplify(tree: BinaryTree) -> BinaryTree:
    nodes_map = {}
    root = None

    for node in tree.postorder_iter():
        attr = tree.nodes[node]
        kind = attr["kind"]

        if kind is Kind.NOT:
            child_id = tree.left(node)
            child_attr = tree.nodes[child_id]
            if child_attr['kind'] is Kind.NOT:
                # ~~a ==> a
                nodes_map[node] = tree.left(child_id)
            else:
                nodes_map[node] = node
        else:
            nodes_map[node] = node

        root = nodes_map[node]

    return map_nodes_of(nodes_map, root, tree)


#
#
# def let(f: PSF, reduce: bool = False, **assignment) -> PSF:
#     logger.debug(f"Applying let operation with {len(assignment)} assignments, reduce={reduce}")
#     if isinstance(f, Terminal):
#         if isinstance(f, Variable):
#             return Constant(assignment[f.value]) if f.value in assignment else f
#         if isinstance(f, BDD):
#             global bdd_manager
#             assign = {v: assignment[v] for v in bdd_manager.support(f.value) if v in assignment}
#             return BDD(bdd_manager.let(assign, f.value))
#
#         return f
#     if isinstance(f, Not):
#         sub_formula = let(f.child, reduce=reduce, **assignment)
#         if isinstance(sub_formula, Constant) and reduce:
#             return Constant(not sub_formula.value)
#
#         return Not(sub_formula)
#     if isinstance(f, And):
#         left_formula = let(f.left_child, reduce=reduce, **assignment)
#         right_formula = let(f.right_child, reduce=reduce, **assignment)
#
#         if isinstance(left_formula, Constant) and left_formula.value == False and reduce:
#             return Constant(False)
#         if isinstance(right_formula, Constant) and right_formula.value == False and reduce:
#             return Constant(False)
#         if isinstance(left_formula, Constant) and isinstance(right_formula, Constant) and reduce:
#             if left_formula.value == True and right_formula.value == True:
#                 return Constant(True)
#
#         return And(let(f.left_child, reduce=reduce, **assignment), let(f.right_child, reduce=reduce, **assignment))
#
#     raise TypeError(f"{type(f)} not recognized.")

@log_perf_counter
def partial_reduce(f: PSF, diagram_size: int, **assignment) -> tuple[PSF, bool]:
    logger.debug(f"Starting partial_reduce with diagram_size={diagram_size}, type={type(f).__name__}, {assignment=}")

    if assignment is None:
        assignment = {}

    builder = Builder()

    nodes_map = {}
    root = None
    last_outcome = False

    for node in nx.dfs_postorder_nodes(f):
        attrs = f.nodes[node]
        kind = attrs['kind']
        if is_terminal(kind):
            value = attrs['value']
            if kind is Kind.CONSTANT:
                manager = create_bdd_manager()
                expr = manager.true if value else manager.false
                node_id = builder.BDD(manager, expr)
                nodes_map[node] = (node_id, True)
            if kind is Kind.VARIABLE:
                manager = create_bdd_manager()
                manager.declare(value)
                expr = manager.add_expr(value)
                if value in assignment:
                    expr = manager.let({value: assignment[value]}, expr)
                node_id = builder.BDD(manager, expr)
                nodes_map[node] = (node_id, True)
            if kind is Kind.CLASS:
                manager = create_bdd_manager()
                manager.declare(value)
                expr = manager.add_expr(value)
                node_id = builder.BDD(manager, expr)
                nodes_map[node] = (node_id, True)
            if kind is Kind.BDD:
                bdd_manager, expr = value
                support = expr.support
                assign = {v: assignment[v] for v in support if v in assignment}
                if assign:
                    expr = bdd_manager.let(assign, expr)

                node_id = builder.BDD(bdd_manager, expr)
                nodes_map[node] = (node_id, expr.dag_size <= diagram_size)
        else:
            if kind is Kind.NOT:
                child_id, outcome = nodes_map[f.left(node)]
                child_attr = builder.T.nodes[child_id]
                if child_attr['kind'] == Kind.BDD:
                    manager, expr = child_attr['value']
                    new_manager, new_expr = Not(manager, expr)
                    node_id = builder.BDD(new_manager, new_expr)

                    nodes_map[node] = (node_id, outcome)
                else:
                    node_id = builder.Not(child_id)
                    nodes_map[node] = (node_id, False)
            elif kind is Kind.AND:
                left_child_id, left_outcome = nodes_map[f.left(node)]
                right_child_id, right_outcome = nodes_map[f.right(node)]

                left_attr = builder.T.nodes[left_child_id]
                right_attr = builder.T.nodes[right_child_id]

                if right_outcome and left_outcome:
                    man_left, f_left = left_attr['value']
                    man_right, f_right = right_attr['value']
                    new_manager, new_expr = And(man_left, f_left, man_right, f_right)
                    node_id = builder.BDD(new_manager, new_expr)

                    nodes_map[node] = (node_id, new_expr.dag_size <= diagram_size)
                else:
                    node_id = builder.And(left_child_id, right_child_id)
                    nodes_map[node] = (node_id, False)
            else:
                logger.critical(f"Node type {kind} not recognized[id={node}]. Skipping...")
                raise TypeError(f"Node type {kind} not recognized[id={node}].")

        root, last_outcome = nodes_map[node]

    built_tree = builder.build()

    return remove_unconnected_nodes(built_tree, root), last_outcome


def psf_feature_counting(f: PSF) -> dict[str, int]:
    feature_count = Counter()

    for leaf in f.leaves:
        leaf_attr = f.nodes[leaf]
        if leaf_attr['kind'] is Kind.BDD:
            _, value = leaf_attr['value']
            feature_count += Counter(features_counting(value))

    return feature_count


def best_feature(f: PSF) -> tuple[str, int]:
    feature_count = psf_feature_counting(f)
    if not feature_count:
        return "", -1

    max_id = max(feature_count, key=lambda x: feature_count[x])
    return max_id, feature_count[max_id]


# def bdd_best_var(f: PSF) -> tuple[str, int]:
#     logger.debug(f"Finding best variable in BDD for type {type(f).__name__}")
#     if isinstance(f, Terminal):
#         if isinstance(f, BDD):
#             return max_occ_var(f.value)
#         return "", 0
#     if isinstance(f, UnaryOperator):
#         return bdd_best_var(f.child)
#     if isinstance(f, BinaryOperator):
#         left_var, left_occurrences = bdd_best_var(f.left_child)
#         right_var, right_occurrences = bdd_best_var(f.right_child)
#         if left_occurrences >= right_occurrences:
#             return left_var, left_occurrences
#
#         return right_var, right_occurrences
#
#     raise TypeError(f"{type(f)} not recognized.")
#
#
@log_perf_counter
def tableau_method(f: PSF) -> TableauTree:
    config = Config()
    tree = tb.Builder()
    root = tree.add_tree(f)

    frontier = deque([root])
    while frontier:
        current = frontier.pop()
        logger.info(f"Visiting {current}")
        current_tree = tree.T.nodes[current]['tree']

        if is_bdd(current_tree):
            logger.info(f"BDD found")
            continue

        # Find best var
        best_var, best_occ = best_feature(current_tree)

        # Low tree iter
        low_tree, _ = partial_reduce(current_tree, config.diagram_size, **{best_var: False})
        low_id = tree.add_tree(low_tree)
        tree.assign(current, low_id, best_var, False)
        frontier.append(low_id)

        # High tree iter
        high_tree, _ = partial_reduce(current_tree, config.diagram_size, **{best_var: True})
        high_id = tree.add_tree(high_tree)
        tree.assign(current, high_id, best_var, True)
        frontier.append(high_id)

    return tree.build()

#     logger.debug(f"Applying tableau method on PSF type {type(f).__name__}")
#     if not tree:
#         logger.info("Creating new tableau tree")
#         current_node = current_node or TableauNode(f)
#         tree = TableauTree(current_node)
#     if current_node.is_bdd():
#         logger.debug("Current node is BDD, returning tree")
#         return tree
#
#     best_var = None
#     best_var_occ = 0
#     for leaf in tree.root.leaves:
#         leaf_var, leaf_occ = bdd_best_var(leaf.psf)
#         if best_var_occ < leaf_occ:
#             best_var = leaf_var
#             best_var_occ = leaf_occ
#
#     logger.debug(f"Best variable found: {best_var} with {best_var_occ} occurrences")
#
#     config_cls = Config()
#     best_var_false = {best_var: False}
#     logger.debug(f"Processing low branch with {best_var}=False")
#     low_psf, _ = partial_reduce(current_node.psf, config_cls.diagram_size, **best_var_false)
#     low_node = TableauNode(low_psf, current_node, best_var_false)
#     tableau_method(low_psf, tree, low_node)
#
#     best_var_true = {best_var: True}
#     logger.debug(f"Processing high branch with {best_var}=True")
#     high_psf, _ = partial_reduce(current_node.psf, config_cls.diagram_size, **best_var_true)
#     high_node = TableauNode(high_psf, current_node, best_var_true)
#     tableau_method(high_psf, tree, high_node)
#
#     logger.debug("Tableau method completed for current node")
#     return tree
