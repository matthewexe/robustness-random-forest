from __future__ import annotations

from robustness.domain.logging import get_logger
from robustness.domain.psf.model import PSF, is_bdd
from robustness.domain.tree.model import BinaryTree

logger = get_logger(__name__)


class Builder:
    current_tree: TableauTree
    next_id: int

    def __init__(self) -> None:
        self.current_tree = TableauTree()
        self.next_id = 0

    def assign(self, current: int, child: int, var: str, assignment: bool):
        self.current_tree.add_edge(
            current, child, var=var, assignment=assignment, label=f"{var}={assignment}"
        )

    def add_psf(self, new_tree: BinaryTree, best_var: str) -> int:
        node_id = self.next_id

        self.current_tree.add_node(
            node_id,
            tree=new_tree,
            best_var=best_var,
            label=f"{node_id} - var:{best_var}" if not is_bdd(new_tree) else f"{node_id} - OBDD",
        )
        self.next_id += 1
        return node_id

    def build(self) -> TableauTree:
        return self.current_tree


class TableauTree(BinaryTree):

    def assignment_of(self, node_id) -> dict[str, int]:
        assignment = {}
        current = node_id
        parent = self.parent(current)

        while parent is not None:
            var, _assignment = self.get_edge_assignment(parent, current)
            assignment[var] = _assignment

            current = parent
            parent = self.parent(current)

        return assignment

    def get_psf_of(self, node_id: int) -> PSF:
        return self.nodes[node_id]["tree"]

    def get_edge_assignment(self, _from: int, _to: int) -> tuple[str, bool]:
        edges_data = self.edges[_from, _to]
        return edges_data["var"], edges_data["assignment"]


def get_node_assignments(tree: BinaryTree, node_id: int) -> dict[str, bool]:
    assignments = {}
    current = node_id

    while tree.in_degree[current] > 0:
        parent = tree.parent(current)
        assignments.update(tree.get_edge_data(parent, current))
        current = parent

    return assignments
