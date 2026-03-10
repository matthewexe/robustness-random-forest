from __future__ import annotations

import networkx as nx

from robustness.domain.bdd import DD_Function, DD_Manager
from robustness.domain.bdd.manager import get_bdd_manager
from robustness.domain.logging import get_logger
from robustness.domain.nx_wrapper import BaseDiGraph

logger = get_logger(__name__)


class BddDag(BaseDiGraph):
    """
    Networkx directed graph wrapper that represents an OBDD
    """

    def __init__(self, **attr: object):
        super().__init__(**attr)

    @staticmethod
    def true() -> int:
        return int(get_bdd_manager().true)

    @staticmethod
    def false() -> int:
        return int(get_bdd_manager().false)

    def add_node(self, node_for_adding: int, **attrs):
        label = attrs['label'] or ""
        del attrs['label']
        super().add_node(node_for_adding, label=f"{label}[{node_for_adding}]", **attrs)

    def set_edge_weight(self, parent: int, child: int, weight: float):
        if self.has_edge(parent, child):
            self.edges[parent, child]['weight'] = weight
            self.edges[parent, child]['label'] = f"{self.edges[parent, child]['label'].split('[')[0]}[c={weight}]"
        else:
            logger.warning(f"Trying to set weight of non existing edge ({parent}, {child})")

    def add_indexed_edge(self, parent: int, child: int, index: int, weight: float = 1, label: str = ""):
        edges_to_remove = set()
        for edge in self.out_edges(parent):
            edge_parent, edge_child = edge
            edge_index = self.edges[edge_parent, edge_child]['index']
            if edge_index == index:
                # self.remove_edge(edge_parent, edge_child)
                edges_to_remove.add((edge_parent, edge_child))

        for (p, c) in edges_to_remove:
            self.remove_edge(p, c)

        self.add_edge(parent, child, weight=weight, index=index, label=f"{label}[c={weight}]")

    def set_low(self, parent: int, child: int, weight: float = 1) -> None:
        self.add_indexed_edge(parent, child, index=0, weight=weight, label="low")

    def set_high(self, parent: int, child: int, weight: float = 1) -> None:
        self.add_indexed_edge(parent, child, index=1, weight=weight, label="high")

    def get_var_of(self, node_id: int) -> str:
        return self.nodes[node_id]['value']

    def get_indexed_child(self, parent: int, index: int) -> int | None:
        for edge in self.out_edges(parent):
            edge_parent, edge_child = edge
            edge_index = self.edges[edge_parent, edge_child]['index']
            if edge_index == index:
                return edge_child

        return None

    def low(self, node: int) -> int | None:
        return self.get_indexed_child(node, 0)

    def high(self, node: int) -> int | None:
        return self.get_indexed_child(node, 1)

    def is_terminal(self, node_id: int) -> bool:
        return self.out_degree[node_id] == 0

    @property
    def root(self) -> int | None:
        for node in self.nodes:
            if self.in_degree[node] == 0:
                return node

        return None


class BddDagBuilder:
    """
    Helper class for building a BDD DAG.
    """

    def __init__(self, manager: DD_Manager):
        self._graph = BddDag()
        self._nodes = {int(manager.true), int(manager.false)}

        self._graph.add_node(int(manager.true), value=True, label="True")
        self._graph.add_node(int(manager.false), value=False, label="False")

    def new_node_from_bdd(self, bdd: DD_Function, **attrs) -> int:
        if int(bdd) in self._nodes:
            return int(bdd)

        node_id = int(bdd)
        self._graph.add_node(int(bdd), label=str(bdd.var), value=bdd.var, **attrs)
        self._nodes.add(int(bdd))
        return node_id

    def set_low(self, parent: int, child: int, weight: float = 1.0) -> BddDagBuilder:
        self._graph.set_low(parent, child, weight)
        return self

    def set_high(self, parent: int, child: int, weight: float = 1.0) -> BddDagBuilder:
        self._graph.set_high(parent, child, weight)
        return self

    def set_children(
            self,
            parent: int,
            low: int | None = None,
            high: int | None = None,
            weight_low: float = 1.0,
            weight_high: float = 1.0,
    ) -> BddDagBuilder:
        if low is not None:
            self._graph.set_low(parent, low, weight_low)
        if high is not None:
            self._graph.set_high(parent, high, weight_high)
        return self

    def build(self) -> BddDag:
        return self._graph
