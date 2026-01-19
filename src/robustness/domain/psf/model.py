from __future__ import annotations

from enum import Enum
from typing import TypeAlias

from robustness.domain.bdd import DD_Function, DD_Manager
from robustness.domain.bdd.manager import get_id_of_manager
from robustness.domain.logging import get_logger
from robustness.domain.tree.model import BinaryTree

PSF: TypeAlias = BinaryTree


class Kind(Enum):
    CONSTANT = "Constant"
    VARIABLE = "Variable"
    CLASS = "Class"
    BDD = "BDD"
    NOT = "Not"
    AND = "And"


logger = get_logger(__name__)


def int_generator(start=0):
    i = start
    while True:
        yield i
        i += 1


class Builder:
    def __init__(self):
        self.T = BinaryTree()
        self._next_id = 0

    # ────────────── core ──────────────

    def _new_node(self, **attrs) -> int:
        nid = self._next_id
        self._next_id += 1
        attrs["label"] = f'{nid} - {attrs.get("label", "")}'
        self.T.add_node(nid, **attrs)
        return nid

    # ────────────── terminals ──────────────

    def Terminal(self, kind: Kind, value, label: str = None) -> int:
        return self._new_node(
            kind=kind,
            value=value,
            label=label
        )

    def Constant(self, value: bool) -> int:
        return self.Terminal(Kind.CONSTANT, value, label=str(value))

    def Variable(self, name: str) -> int:
        return self.Terminal(Kind.VARIABLE, name, label=f"Var({name})")

    def Class(self, name: str) -> int:
        return self.Terminal(Kind.CLASS, name, label=f"Class({name})")

    def BDD(self, manager: DD_Manager, value: DD_Function) -> int:
        return self.Terminal(Kind.BDD, (manager, value), label=f"BDD({get_id_of_manager(manager)}_{int(value)})")

    # ────────────── operators ──────────────

    def Not(self, child: int) -> int:
        n = self._new_node(kind=Kind.NOT, label="Not")
        self.T.add_left(n, child)
        return n

    def And(self, left: int, right: int) -> int:
        n = self._new_node(kind=Kind.AND, label="And")
        self.T.add_left(n, left)
        self.T.add_right(n, right)
        return n

    def Or(self, left: int, right: int) -> int:
        # De Morgan
        return self.Not(self.And(self.Not(left), self.Not(right)))

    # ────────────── accessors ──────────────

    def left(self, node: int) -> int | None:
        return self.T.left(node)

    def right(self, node: int) -> int | None:
        return self.T.right(node)

    def parent(self, node: int) -> int | None:
        return self.T.parent(node)

    # ────────────── export ──────────────

    def build(self) -> BinaryTree:
        return self.T


def is_terminal(kind: Kind):
    return kind in {Kind.VARIABLE, Kind.CONSTANT, Kind.CLASS, Kind.BDD}


def is_bdd(f: PSF):
    if not f.nodes or len(f.nodes) > 1:
        return False

    root_attr = f.nodes[f.root]
    return root_attr['kind'] is Kind.BDD


def render_formula(f: PSF):
    memo = {}

    for node in f.postorder_iter():
        attr = f.nodes[node]

        kind = attr['kind']
        if is_terminal(kind):
            if kind is Kind.BDD:
                memo[node] = f"{get_id_of_manager(attr['value'][0])}_{int(attr['value'][1])}"
            else:
                memo[node] = str(attr['value'])
        else:
            if kind is Kind.NOT:
                child = f.left(node)
                memo[node] = f"~({memo[child]})"
            elif kind is Kind.AND:
                left_child = f.left(node)
                right_child = f.right(node)
                memo[node] = f"{memo[left_child]} & {memo[right_child]}"

    return memo[f.root]
