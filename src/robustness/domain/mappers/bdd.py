import networkx as nx

from robustness.domain.bdd import DD_Function, DD_Manager
from robustness.domain.bdd.dag import BddDag, BddDagBuilder
from robustness.domain.bdd.manager import create_bdd_manager
from robustness.domain.bdd.operations import path_of
from robustness.domain.logging import get_logger
from robustness.domain.random_forest import Sample, Endpoints
from robustness.domain.utils.formula import is_class, get_class_label

_manager = create_bdd_manager()
logger = get_logger(__name__)


def to_dag(manager: DD_Manager, bdd: DD_Function) -> BddDag:
    b = BddDagBuilder(manager)

    def rec(root: DD_Function) -> int:
        if root.var is None:  # is true or false
            return int(root)

        low_node = rec(root.low)
        high_node = rec(root.high)
        new_node = b.new_node_from_bdd(root)
        b.children(new_node, low=low_node, high=high_node)
        return new_node

    rec(bdd)
    return b.build()


def construct_robustness_dag(manager: DD_Manager, bdd: DD_Function, sample: Sample, path: str):
    dag = to_dag(manager, bdd)

    current = dag.root
    while not dag.is_terminal(current):
        attr = dag.nodes[current]
        choice, path = path[-1], path[:-1]

        if not path:
            break
        value = attr['value']
        if is_class(value) and sample.predicted_label == get_class_label(value):
            continue

        if choice == "0":
            next_node = dag.low(current)
            dag.remove_edge(current, next_node)
            dag.set_low(current, next_node, weight=0)
        else:
            next_node = dag.high(current)
            dag.remove_edge(current, next_node)
            dag.set_high(current, next_node, weight=0)

        current = next_node

    # Set in edges weight of False node to infinity
    import math
    for u, v, data in dag.in_edges(dag.false(), data=True):
        dag.add_indexed_edge(u, v, index=data['index'], weight=math.inf, label=data['label'])

    return remove_class_from_dag(sample.predicted_label, dag)


def remove_class_from_dag(class_label: str, dag: BddDag) -> BddDag:
    new_dag = dag.copy()

    edges_to_remove = set()

    for node in new_dag.nodes:
        attr = new_dag.nodes[node]
        value = attr['value']
        if not is_class(value) or get_class_label(value) != class_label:
            continue

        high_child = new_dag.high(node)
        edges_to_remove.add((node, high_child))

    for edge in edges_to_remove:
        parent, child = edge
        new_dag.remove_edge(parent, child)

    return new_dag


def calculate_robustness(manager: DD_Manager, bdd: DD_Function, sample: Sample, endpoints: Endpoints) -> float:
    if bdd == manager.true:
        return 0
    elif bdd == manager.false:
        import math
        return math.inf

    path = path_of(sample, bdd, endpoints)
    dag = construct_robustness_dag(bdd, sample, path)
    shortest_path = nx.shortest_path(dag, dag.root, dag.true())
    logger.info(f"Shortest path: {", ".join(map(str, shortest_path))}")

    # Sum weight of all edges along the path
    path_weight = 0
    for i in range(len(shortest_path) - 1):
        u, v = shortest_path[i], shortest_path[i + 1]
        if dag.has_edge(u, v):
            path_weight += dag[u][v].get('weight', 1)
        else:
            raise ValueError(f"Edge ({u}, {v}) does not exist in the graph.")

    return path_weight
