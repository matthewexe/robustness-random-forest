from __future__ import annotations

import networkx as nx

from robustness.domain.bdd import DD_Function, DD_Manager
from robustness.domain.logging import get_logger

logger = get_logger(__name__)


class BddDag(nx.DiGraph):

    def __init__(self, **attr: object):
        super().__init__(**attr)

    @staticmethod
    def true() -> int:
        return 1

    @staticmethod
    def false() -> int:
        return -1

    def add_node(self, node_for_adding: int, **attrs):
        label = attrs['label'] or ""
        del attrs['label']
        super().add_node(node_for_adding, label=f"{label}[{node_for_adding}]", **attrs)

    def add_indexed_edge(self, parent: int, child: int, index: int, weight: float = 1, label: str = "", **attrs):
        for edge in self.out_edges(parent):
            edge_parent, edge_child = edge
            edge_index = self.edges[edge_parent, edge_child]['index']
            if edge_index == index and child != edge_child:
                logger.warning(f"Switch indexed edge {index} of {parent} node with {child} child node")
                self.remove_edge(edge_parent, edge_child)

        self.add_edge(parent, child, weight=weight, index=index, label=f"{label}[{weight}]")

    def set_low(self, parent: int, child: int, weight: float = 1) -> None:
        self.add_indexed_edge(parent, child, index=0, weight=weight, label="low")

    def set_high(self, parent: int, child: int, weight: float = 1) -> None:
        self.add_indexed_edge(parent, child, index=1, weight=weight, label="high")

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
    def __init__(self, manager: DD_Manager):
        self._graph = BddDag()
        self._nodes = {int(manager.true), int(manager.false)}

        self._graph.add_node(int(manager.true), value=True, label="True")
        self._graph.add_node(int(manager.false), value=False, label="False")

    # ---------- node creation ----------

    def new_node_from_bdd(self, bdd: DD_Function, **attrs) -> int:
        if int(bdd) in self._nodes:
            return int(bdd)

        node_id = int(bdd)
        self._graph.add_node(int(bdd), label=str(bdd.var), value=bdd.var, **attrs)
        self._nodes.add(int(bdd))
        return node_id

    def terminal(self, value) -> int:
        return self.new_node(value=value)

    # ---------- edge wiring ----------

    def low(self, parent: int, child: int, weight: float = 1.0) -> BddDagBuilder:
        self._graph.set_low(parent, child, weight)
        return self

    def high(self, parent: int, child: int, weight: float = 1.0) -> BddDagBuilder:
        self._graph.set_high(parent, child, weight)
        return self

    def children(
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

    # ---------- access ----------

    def build(self) -> BddDag:
        return self._graph
