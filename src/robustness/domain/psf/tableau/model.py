from __future__ import annotations

from robustness.domain.logging import get_logger
from robustness.domain.psf.model import render_formula
from robustness.domain.tree.model import BinaryTree

logger = get_logger(__name__)

class Builder:
    T: TableauTree
    next_id: int

    def __init__(self) -> None:
        self.T = TableauTree()
        self.next_id = 0

    def assign(self, current: int, child: int, var: str, value: bool):
        self.T.add_edge(current, child, var=var, value=value)

    def add_tree(self, new_tree: BinaryTree) -> int:
        node_id = self.next_id
        formula = render_formula(new_tree)
        if len(formula) > 200:
            formula = formula[:197] + "..."

        self.T.add_node(node_id, tree=new_tree, label=formula)
        self.next_id += 1
        return node_id

    def build(self) -> TableauTree:
        return self.T

class TableauTree(BinaryTree):

    def assignment_of(self, node_id) -> dict[str, int]:
        assignment = {}
        current = node_id
        parent = self.parent(current)

        while parent is not None:
            data = self.get_edge_data(parent, current)
            assignment[data['var']] = data['value']

            current = parent
            parent= self.parent(current)

        return assignment


def get_node_assignments(tree: BinaryTree, node_id: int) -> dict[str, bool]:
    assignments = {}
    current = node_id

    while tree.in_degree[current] > 0:
        parent = tree.parent(current)
        assignments.update(tree.get_edge_data(parent, current))
        current = parent

    return assignments
