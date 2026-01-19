from __future__ import annotations

import os.path
from collections import deque, Counter
from os import PathLike

import networkx as nx
from networkx.drawing.nx_agraph import write_dot

import robustness.domain.psf.tableau.model as tb
from robustness.domain.bdd.manager import create_bdd_manager
from robustness.domain.bdd.operations import features_counting, Not, And, write_bdd
from robustness.domain.config import Config
from robustness.domain.logging import get_logger
from robustness.domain.mappers.bdd import calculate_robustness
from robustness.domain.psf.model import PSF, Kind, Builder, is_terminal, is_bdd, render_formula
from robustness.domain.psf.tableau.model import TableauTree
from robustness.domain.random_forest import Sample, Endpoints
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


@log_perf_counter
def partial_reduce(psf: PSF, diagram_size: int, **assignment) -> tuple[PSF, bool]:
    config = Config()

    if not os.path.isdir("logs/bdds") and config.debug_mode:
        os.makedirs("logs/bdds", exist_ok=True)

    logger.debug(f"Starting partial_reduce with diagram_size={diagram_size}, type={type(psf).__name__}, {assignment=}")

    if assignment is None:
        assignment = {}

    builder = Builder()

    nodes_map = {}
    root = None
    last_outcome = False

    for node in nx.dfs_postorder_nodes(psf):
        attrs = psf.nodes[node]
        kind = attrs['kind']
        if is_terminal(kind):
            value = attrs['value']
            if kind is Kind.CONSTANT:
                manager, _ = create_bdd_manager()
                expr = manager.true if value else manager.false
                node_id = builder.BDD(manager, expr)
                nodes_map[node] = (node_id, True)
            if kind is Kind.VARIABLE:
                manager, new_id_manager = create_bdd_manager()
                manager.declare(value)
                expr = manager.add_expr(value)
                if value in assignment:
                    expr = manager.let({value: assignment[value]}, expr)
                node_id = builder.BDD(manager, expr)

                # Write BDD to file if in debug mode
                if config.log_graphs:
                    write_bdd(manager, expr, new_id_manager)

                nodes_map[node] = (node_id, True)
            if kind is Kind.CLASS:
                manager, new_id_manager = create_bdd_manager()
                manager.declare(value)
                expr = manager.add_expr(value)

                # write BDD to file if in debug mode
                if config.log_graphs:
                    write_bdd(manager, expr, new_id_manager)

                node_id = builder.BDD(manager, expr)
                nodes_map[node] = (node_id, True)
            if kind is Kind.BDD:
                bdd_manager, expr = value
                support = expr.support
                assign = {v: assignment[v] for v in support if v in assignment}
                if assign:
                    expr = bdd_manager.let(assign, expr)

                node_id = builder.BDD(bdd_manager, expr)

                # write BDD to file in debug mode
                if config.log_graphs:
                    write_bdd(bdd_manager, expr)

                nodes_map[node] = (node_id, expr.dag_size <= diagram_size)
        else:
            if kind is Kind.NOT:
                child_id, outcome = nodes_map[psf.left(node)]
                child_attr = builder.T.nodes[child_id]
                if child_attr['kind'] == Kind.BDD:
                    manager, expr = child_attr['value']
                    new_manager, new_expr, new_id_manager = Not(manager, expr)
                    node_id = builder.BDD(new_manager, new_expr)

                    if config.log_graphs:
                        write_bdd(new_manager, new_expr, new_id_manager)

                    nodes_map[node] = (node_id, outcome)
                else:
                    node_id = builder.Not(child_id)
                    nodes_map[node] = (node_id, False)
            elif kind is Kind.AND:
                left_child_id, left_outcome = nodes_map[psf.left(node)]
                right_child_id, right_outcome = nodes_map[psf.right(node)]

                left_attr = builder.T.nodes[left_child_id]
                right_attr = builder.T.nodes[right_child_id]

                if right_outcome and left_outcome:
                    man_left, f_left = left_attr['value']
                    man_right, f_right = right_attr['value']
                    new_manager, new_expr, new_id_manager = And(man_left, f_left, man_right, f_right)
                    node_id = builder.BDD(new_manager, new_expr)

                    if config.log_graphs:
                        write_bdd(new_manager, new_expr, new_id_manager)

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


@log_perf_counter
def tableau_method(f: PSF) -> TableauTree:
    config = Config()

    if not os.path.isdir("logs/tableau") and config.debug_mode:
        os.makedirs("logs/tableau", exist_ok=True)

    tree = tb.Builder()
    root = tree.add_tree(f, "None")
    if config.log_graphs:
        write_psf(f, f"{root}_root.dot")

    frontier = deque([root])
    while frontier:
        current = frontier.pop()
        logger.info(f"Visiting {current}")
        current_tree = tree.T.nodes[current]['tree']

        if is_bdd(current_tree):
            logger.info(f"BDD found")
            if config.log_graphs:
                write_psf(current_tree, f"{current}_bdd.dot")
            continue

        # Find best var
        best_var, best_occ = best_feature(current_tree)

        # Low tree iter
        low_tree, _ = partial_reduce(current_tree, config.diagram_size, **{best_var: False})
        low_id = tree.add_tree(low_tree, best_feature(low_tree)[0])
        tree.assign(current, low_id, best_var, False)
        frontier.append(low_id)

        if config.log_graphs:
            write_psf(low_tree, f"{low_id}_low_{current}_{best_var}.dot")

        # High tree iter
        high_tree, _ = partial_reduce(current_tree, config.diagram_size, **{best_var: True})
        high_id = tree.add_tree(high_tree, best_feature(high_tree)[0])
        tree.assign(current, high_id, best_var, True)
        frontier.append(high_id)

        if config.log_graphs:
            write_psf(high_tree, f"{high_id}_high_{current}_{best_var}.dot")

    build = tree.build()
    if config.log_graphs:
        write_dot(build, os.path.join("logs/tableau", "final_tableau.dot"))

    return build


def robustness(t: TableauTree, sample: Sample, endpoints: Endpoints) -> int:
    memo = {}

    for leaf in t.leaves:
        leaf_tree = t.nodes[leaf]['tree']
        leaf_tree_root = leaf_tree.root
        manager, bdd = leaf_tree.nodes[leaf_tree_root]['value']
        memo[leaf] = calculate_robustness(manager, bdd, sample, endpoints)
        parent = t.parent(leaf)
        current = leaf

        # Calculate path cost to root
        while parent is not None:
            edge_data = t.edges[parent, current]
            var = edge_data['var']
            path_cost = memo[current]
            if not edge_data['value']:
                # Low branch
                path_cost += 0 if sample.features[var] <= endpoints[var] else 1
            else:
                # High branch
                path_cost += 0 if int(sample.features[var] > endpoints[var]) else 1

            memo[parent] = min(memo.get(parent, float('inf')), path_cost)

            current = parent
            parent = t.parent(current)

    return memo[t.root]

def generate_robustness_graph(t: TableauTree, sample: Sample, endpoints: Endpoints, filename: PathLike = "robustness_report.dot"):
    memo_nodes = {}
    memo_edges = {}

    for leaf in t.leaves:
        leaf_tree = t.nodes[leaf]['tree']
        leaf_tree_root = leaf_tree.root
        attrs = leaf_tree.nodes[leaf_tree_root]
        manager, bdd = attrs['value']
        path_cost = calculate_robustness(manager, bdd, sample, endpoints)
        parent = t.parent(leaf)
        current = leaf
        memo_nodes[leaf] = f"{attrs['label']} - Cost: {path_cost}"

        # Calculate path cost to root
        while parent is not None:
            edge_data = t.edges[parent, current]
            var = edge_data['var']
            if not edge_data['value']:
                # Low branch
                path_cost = 0 if sample.features[var] <= endpoints[var] else 1
            else:
                # High branch
                path_cost = 0 if int(sample.features[var] > endpoints[var]) else 1

            memo_nodes[parent] = f"{t.nodes[parent]['best_var']}"
            memo_edges[(parent, current)] = f"{path_cost}"

            current = parent
            parent = t.parent(current)

    # Create a new graph to represent the robustness paths
    import graphviz
    robustness_graph = graphviz.Digraph(comment="Robustness Graph", format="dot")
    for node, label in memo_nodes.items():
        robustness_graph.node(str(node), label=str(label))

    for (src, dst), label in memo_edges.items():
        robustness_graph.edge(str(src), str(dst), label=str(label))

    # Write as SVG
    path_root, _ = os.path.splitext(filename)
    svg_filename = f"{path_root}.svg"
    path = os.path.join("logs", svg_filename)
    os.makedirs(os.path.dirname(path) or "logs", exist_ok=True)
    svg_bytes = robustness_graph.pipe(format="svg")
    with open(path, "wb") as f:
        f.write(svg_bytes)

def write_psf(psf: PSF, filename: str) -> None:
    config = Config()
    logger.debug(f"Writing PSF to logs/psf/{filename}: {render_formula(psf)}")
    if not os.path.isdir("logs/psf") and config.debug_mode:
        os.makedirs("logs/psf", exist_ok=True)

    path = os.path.join("logs/psf", filename)
    write_dot(psf, path)
