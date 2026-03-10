import math
from collections import deque

import networkx as nx

from robustness.domain.bdd import DD_Function, DD_Manager
from robustness.domain.bdd.dag import BddDag, BddDagBuilder
from robustness.domain.bdd.manager import get_bdd_manager
from robustness.domain.bdd.operations import test_sample
from robustness.domain.config import Config
from robustness.domain.logging import get_logger
from robustness.domain.random_forest import Sample, Endpoints
from robustness.domain.utils.formula import is_class, get_class_label

logger = get_logger(__name__)
config = Config()


def to_dag(manager: DD_Manager, bdd: DD_Function) -> BddDag:
    b = BddDagBuilder(manager)

    def rec(root: DD_Function) -> int:
        if root.var is None:  # is true or false
            return int(root)

        low_node = rec(root.low)
        high_node = rec(root.high)
        new_node = b.new_node_from_bdd(root)
        b.set_children(new_node, low=low_node, high=high_node)
        return new_node

    rec(bdd)
    return b.build()


def construct_robustness_dag(manager: DD_Manager, bdd: DD_Function, sample: Sample, endpoints: Endpoints) -> BddDag:
    dag = to_dag(manager, bdd)

    frontier = deque()
    frontier.append(dag.root)
    # DFS
    while frontier:
        current = frontier.pop()
        if current in (dag.false(), dag.true()):
            continue

        for child in dag.successors(current):
            frontier.append(child)

        var = dag.get_var_of(current)

        if is_class(var):
            # Remove high edge from predicted class nodes
            if get_class_label(var) == sample.predicted_label:
                current_child = dag.high(current)
                if current_child is not None:  # Could be None if it's deleted in previous iterations
                    dag.set_high(current, current_child, weight=math.inf)
            else:  # Set not predicted class low edges as 0 weighted edge
                current_child = dag.low(current)
                dag.set_low(current, current_child, weight=0)
        else:
            # Set 0 weighted edges
            if sample.features[var] <= endpoints[var]:
                current_child = dag.low(current)
                # dag.remove_edge(current, current_child)
                dag.set_low(current, current_child, weight=0)
            else:
                current_child = dag.high(current)
                # dag.remove_edge(current, current_child)
                dag.set_high(current, current_child, weight=0)

    # Set in edges weight of False node to infinity
    for u, v, data in dag.in_edges(dag.false(), data=True):
        dag.set_edge_weight(u, v, weight=math.inf)

    return dag


def calculate_bdd_robustness(f: DD_Function, sample: Sample, endpoints: Endpoints) -> float:
    """
    Calculates the robustness of a BDD function given a sample.

    Args:
        manager: the BDD manager
        f: the BDD function to analyze
        sample: the sample to test
        endpoints: the endpoints universe to test the sample against

    Returns: robustness value

    """
    manager = get_bdd_manager()
    if f == manager.true:
        return 0
    elif f == manager.false:
        import math
        return math.inf

    dag = construct_robustness_dag(manager, f, sample, endpoints)

    if config.log_graphs:
        dag.save_svg(f"logs/robustness/bdd_dags/{int(f)}_robustness_dag.svg")

    shortest_path = nx.shortest_path(dag, dag.root, dag.true(), weight="weight")
    logger.debug(f"Shortest path for {int(f)}: {", ".join(map(str, shortest_path))}")

    # Sum weight of all edges along the path
    path_weight = 0
    for i in range(len(shortest_path) - 1):
        u, v = shortest_path[i], shortest_path[i + 1]
        if dag.has_edge(u, v):
            path_weight += dag[u][v].get('weight', 1)
        else:
            raise ValueError(f"Edge ({u}, {v}) does not exist in the graph.")

    logger.info(f"Calculated robustness for BDD {int(f)}: {shortest_path}")

    return path_weight
