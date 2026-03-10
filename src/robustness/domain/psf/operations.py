from __future__ import annotations

import os.path
from collections import deque, Counter
from os import PathLike

import robustness.domain.bdd.operations as bdd_ops
import robustness.domain.psf.tableau.model as tb
from robustness.domain.bdd.manager import get_bdd_manager
from robustness.domain.config import Config
from robustness.domain.logging import get_logger
from robustness.domain.mappers.bdd import calculate_bdd_robustness
from robustness.domain.psf.model import PSF, Kind, Builder, is_bdd, render_formula
from robustness.domain.psf.tableau.model import TableauTree
from robustness.domain.random_forest import Sample, Endpoints
from robustness.domain.tree.operations import remove_unreachable_nodes_from
from robustness.domain.utils.metrics import log_perf_counter

logger = get_logger(__name__)


@log_perf_counter
def partial_reduce(
        psf: PSF, diagram_size: int, assignment: dict[str, bool] | None = None
) -> tuple[PSF, bool]:
    """
    Try to reduce the PSF formula by applying the given variable assignment and
    converting subformulas to BDDs when their size doesn't exceed diagram_size.

    Args:
        psf: PSF formula to reduce.
        diagram_size: Maximum allowed size for BDD conversion.
        **assignment: Variable assignment to apply to PSF.

    Returns: Reduced PSF formula and boolean indicating if it could be reduced more

    """
    config = Config()

    if not os.path.isdir("logs/bdds") and config.debug_mode:
        os.makedirs("logs/bdds", exist_ok=True)

    logger.debug(
        f"Starting partial_reduce with diagram_size={diagram_size}, type={type(psf).__name__}, {assignment=}"
    )

    if assignment is None:
        assignment = {}

    builder = Builder()

    nodes_map = {}
    root = None
    last_outcome = False

    # Iterate in postorder to ensure children are processed before parents and simulate a stack-based evaluation
    for psf_node_id in psf.postorder_iter():
        # attrs = psf.get_node_attrs(psf_node_id)
        kind = psf.get_kind_of(psf_node_id)
        match kind:
            case Kind.CONSTANT:
                constant: bool = psf.get_value_of(psf_node_id)
                manager = get_bdd_manager()
                expr = manager.true if constant else manager.false
                node_id = builder.BDD(expr)
                nodes_map[psf_node_id] = (node_id, True)
            case Kind.VARIABLE:
                variable_label: str = psf.get_value_of(psf_node_id)
                manager = get_bdd_manager()
                manager.declare(variable_label)
                expr = manager.add_expr(variable_label)
                if variable_label in assignment:
                    expr = manager.true if assignment[variable_label] else manager.false
                node_id = builder.BDD(expr)

                if config.log_graphs:
                    bdd_ops.write_bdd(manager, expr)

                nodes_map[psf_node_id] = (node_id, True)
            case Kind.CLASS:
                class_label: str = psf.get_value_of(psf_node_id)
                manager = get_bdd_manager()
                manager.declare(class_label)
                expr = manager.add_expr(class_label)

                if config.log_graphs:
                    bdd_ops.write_bdd(manager, expr)

                node_id = builder.BDD(expr)
                nodes_map[psf_node_id] = (node_id, True)
            case Kind.BDD:
                bdd_manager = get_bdd_manager()
                expr = psf.get_value_of(psf_node_id)
                assign = {v: assignment[v] for v in expr.support if v in assignment}
                if len(assign) > 0:
                    expr = bdd_manager.let(assign, expr)

                node_id = builder.BDD(expr)

                if config.log_graphs:
                    bdd_ops.write_bdd(bdd_manager, expr)

                nodes_map[psf_node_id] = (node_id, expr.dag_size <= diagram_size)
            case Kind.NOT:
                child_id, outcome = nodes_map[psf.left(psf_node_id)]
                child_kind = builder.current_psf.get_kind_of(child_id)
                child_value = builder.current_psf.get_value_of(child_id)
                if child_kind == Kind.BDD:
                    manager = get_bdd_manager()
                    expr = child_value
                    new_expr = manager.apply("not", expr)
                    node_id = builder.BDD(new_expr)

                    if config.log_graphs:
                        bdd_ops.write_bdd(manager, new_expr)

                    nodes_map[psf_node_id] = (node_id, outcome)
                else:
                    node_id = builder.Not(child_id)
                    nodes_map[psf_node_id] = (node_id, False)
            case Kind.AND:
                left_child_id, left_outcome = nodes_map[psf.left(psf_node_id)]
                right_child_id, right_outcome = nodes_map[psf.right(psf_node_id)]

                if right_outcome and left_outcome:
                    bdd_manager = get_bdd_manager()
                    # left_attr = builder.current_psf.get_node_attrs(left_child_id)
                    # right_attr = builder.current_psf.get_node_attrs(right_child_id)
                    bdd_left = builder.current_psf.get_value_of(left_child_id)
                    bdd_right = builder.current_psf.get_value_of(right_child_id)
                    new_expr = bdd_manager.apply("and", bdd_left, bdd_right)
                    node_id = builder.BDD(new_expr)

                    if config.log_graphs:
                        bdd_ops.write_bdd(bdd_manager, new_expr)

                    nodes_map[psf_node_id] = (
                        node_id,
                        new_expr.dag_size <= diagram_size,
                    )
                else:
                    node_id = builder.And(left_child_id, right_child_id)
                    nodes_map[psf_node_id] = (node_id, False)
            case _:
                logger.critical(
                    f"Node type {kind} not recognized[id={psf_node_id}]. Aborting..."
                )
                raise RuntimeError(f"Node type {kind} not recognized")

        root, last_outcome = nodes_map[psf_node_id]

    built_tree = builder.build()

    """
    During the building process, some bdds might not be connected due to the fact that 
    they could be reduced more and they will not be connected in future iterations.
    """
    reduced_tree: PSF = remove_unreachable_nodes_from(built_tree, root)
    return reduced_tree, last_outcome


def psf_feature_counting(psf: PSF) -> dict[str, int]:
    """
    Given a PSF formula, count the occurrences of each feature (variable) in the BDDs contained in the formula.

    Args:
        psf: PSF formula to analyze.

    Returns: A dictionary mapping feature names to their occurrence counts in the BDDs of the PSF formula.

    """
    feature_count = Counter()

    for leaf in psf.leaves:
        leaf_kind = psf.get_kind_of(leaf)
        leaf_value = psf.get_value_of(leaf)
        if leaf_kind is Kind.BDD:
            counting = bdd_ops.features_counting(leaf_value)
            feature_count += Counter(counting)

    return feature_count


def best_feature(f: PSF) -> tuple[str, int]:
    """
    Select the most frequent feature in psf bdds. If there are no features, return an empty string and -1.

    Args:
        f: PSF formula to analyze.

    Returns: Most frequent feature and its count.

    """
    feature_count = psf_feature_counting(f)
    if not feature_count:
        return "", -1

    max_id = max(feature_count, key=lambda x: feature_count[x])
    return max_id, feature_count[max_id]


@log_perf_counter
def tableau_method(f: PSF) -> TableauTree:
    """
    Tableau method builder for PSF formulas.

    Args:
        f: PSF formula to build the tableau for.

    Returns: TableauTree

    """
    config = Config()

    if not os.path.isdir("logs/tableau") and config.debug_mode:
        os.makedirs("logs/tableau", exist_ok=True)

    tree = tb.Builder()
    root = tree.add_psf(f, "Initial Reduced-PSF")
    if config.log_graphs:
        log_psf(f, f"{root}_root.svg")

    frontier = deque([root])
    # Using DFS
    while frontier:
        current = frontier.pop()
        logger.debug(f"Visiting {current}")
        current_psf = tree.current_tree.get_psf_of(current)

        if is_bdd(current_psf):
            logger.info(f"BDD found")
            continue

        # Find best var
        best_var, best_occ = best_feature(current_psf)
        logger.info(f"Splitting on variable {best_var} with occurrence {best_occ}")

        # Low tree iter
        low_tree, _ = partial_reduce(
            current_psf, config.diagram_size, {best_var: False}
        )
        low_id = tree.add_psf(low_tree, best_feature(low_tree)[0])
        tree.assign(current, low_id, best_var, False)
        frontier.append(low_id)

        if config.log_graphs:
            log_psf(low_tree, f"{low_id}.svg")

        # High tree iter
        high_tree, _ = partial_reduce(
            current_psf, config.diagram_size, {best_var: True}
        )
        high_id = tree.add_psf(high_tree, best_feature(high_tree)[0])
        tree.assign(current, high_id, best_var, True)
        frontier.append(high_id)

        if config.log_graphs:
            log_psf(high_tree, f"{high_id}.svg")

    build = tree.build()

    return build


def robustness(t: TableauTree, sample: Sample, endpoints: Endpoints) -> int:
    """
    Calculate robustness based on tableau tree and a sample.
    It calculates how many bits we need to flip in the sample to reach a leaf that corresponds to a positive classification.

    Args:
        t: Tableau tree built from a PSF formula.
        sample: Sample to test robustness on.
        endpoints: Endpoints universe to determine the thresholds for feature values.

    Returns: Robustness value (number of bits to flip).

    """
    memo = {}

    for leaf in t.leaves:
        leaf_psf = t.get_psf_of(leaf)
        bdd = leaf_psf.get_value_of(leaf_psf.root_id)

        memo[leaf] = calculate_bdd_robustness(bdd, sample, endpoints)
        parent = t.parent(leaf)
        current = leaf

        # Calculate path cost to root
        while parent is not None:
            var, assignment = t.get_edge_assignment(parent, current)
            path_cost = memo[current]
            if not assignment:
                # Low branch
                path_cost += 0 if sample.features[var] <= endpoints[var] else 1
            else:
                # High branch
                path_cost += 0 if int(sample.features[var] > endpoints[var]) else 1

            memo[parent] = min(memo.get(parent, float("inf")), path_cost)

            current = parent
            parent = t.parent(current)

    return memo[t.root_id]


def generate_robustness_graph(
        t: TableauTree,
        sample: Sample,
        endpoints: Endpoints,
        filename: PathLike | str = "robustness_report.dot",
) -> None:
    """
    Used to generate a graphviz representation of the tableau tree with cost path based on sample.

    Args:
        t: Tableau tree built from a PSF formula.
        sample: Sample to test robustness on.
        endpoints: Endpoints universe to determine the thresholds for feature values.
        filename: Name of the file to write the graph to.

    Returns: None
    """

    memo_nodes = {}
    memo_edges = {}

    for leaf in t.leaves:
        leaf_psf = t.get_psf_of(leaf)
        leaf_root_id = leaf_psf.root_id
        bdd = leaf_psf.get_value_of(leaf_root_id)
        path_cost = calculate_bdd_robustness(bdd, sample, endpoints)
        parent = t.parent(leaf)
        current = leaf
        label = leaf_psf.get_label_of(leaf_root_id)
        memo_nodes[leaf] = f"{label} - Cost: {path_cost}"

        # Calculate path cost to root
        while parent is not None:
            var, assignment = t.get_edge_assignment(parent, current)
            if not assignment:
                # Low branch
                path_cost = 0 if sample.features[var] <= endpoints[var] else 1
            else:
                # High branch
                path_cost = 0 if int(sample.features[var] > endpoints[var]) else 1

            memo_nodes[parent] = f"{t.nodes[parent]['best_var']}"
            memo_edges[(parent, current)] = (
                f"{var}={'true' if assignment else "false"}\n"
                f"Cost: {path_cost}\n"
                f"sample[{var}]={sample.features[var]:.4f},endpoint={endpoints[var]:.4f}"
            )

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
    path = os.path.join("logs/robustness", svg_filename)
    os.makedirs(os.path.dirname(path) or "logs", exist_ok=True)
    svg_bytes = robustness_graph.pipe(format="svg")
    with open(path, "wb") as f:
        f.write(svg_bytes)


def log_psf(psf: PSF, filename: str) -> None:
    """Used to write a PSF formula as a svg file for visualization and debugging purposes.

    Args:
        psf: PSF formula towrite.
        filename: Name of the file to write the graph to.

    Returns: None
    """

    config = Config()
    logger.debug(f"Writing PSF to logs/psf/{filename}: {render_formula(psf)}")
    if not os.path.isdir("logs/psf") and config.debug_mode:
        os.makedirs("logs/psf", exist_ok=True)

    path = os.path.join("logs/psf", filename)
    psf.save_svg(path)
