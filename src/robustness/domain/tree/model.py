from os import PathLike
from typing import override, Iterator

import networkx as nx


class BinaryTree(nx.DiGraph):
    """
    Networkx directed graph wrapper that represents a binary tree.
    """

    def __init__(self, **attr: object):
        super().__init__(**attr)

    @property
    def root_id(self) -> int | None:
        for n in self.nodes:
            if self.in_degree(n) == 0:
                return n

        return None

    @override
    def add_node(self, node_for_adding, **attr):
        super().add_node(node_for_adding, **attr)

    def parent(self, node_id: int) -> int | None:
        pred = list(self.predecessors(node_id))
        if not pred:
            return None

        return pred[0]

    def left(self, node: int) -> int | None:
        for _, v, d in self.out_edges(node, data=True):
            if d["index"] == 0:
                return v
        return None

    def right(self, node: int) -> int | None:
        for _, v, d in self.out_edges(node, data=True):
            if d["index"] == 1:
                return v
        return None

    def _add_child(self, parent: int, child: int, index: int):
        for _, v, d in self.out_edges(parent, data=True):
            if d["index"] == index:
                raise ValueError("Left child already exists")

        self.add_edge(parent, child, index=index)

    def add_left(self, parent, child):
        self._add_child(parent, child, index=0)

    def add_right(self, parent, child):
        self._add_child(parent, child, index=1)

    def postorder_iter(self):
        return nx.dfs_postorder_nodes(self, self.root_id)

    def get_node_attrs(self, node_id: int) -> dict:
        return self.nodes[node_id]

    def get_label_of(self, node_id: int) -> str:
        return self.nodes[node_id].get("label", "")

    @property
    def leaves(self) -> Iterator[int]:
        return (node for node in self.nodes if self.out_degree[node] == 0)

    def save_svg(self, path: PathLike | str):
        import networkx as nx

        A = nx.nx_agraph.to_agraph(self)
        A.draw(path, format='svg', prog='dot')
