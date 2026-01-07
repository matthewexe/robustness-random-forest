from __future__ import annotations

from enum import Enum
from typing import TypeAlias

from robustness.domain.bdd import DD_Function
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
        self.T.add_node(nid, **attrs)
        return nid

    # ────────────── terminals ──────────────

    def Terminal(self, kind: Kind, value) -> int:
        return self._new_node(
            kind=kind,
            value=value,
        )

    def Constant(self, value: bool) -> int:
        return self.Terminal(Kind.CONSTANT, value)

    def Variable(self, name: str) -> int:
        return self.Terminal(Kind.VARIABLE, name)

    def Class(self, name: str) -> int:
        return self.Terminal(Kind.CLASS, name)

    def BDD(self, value: DD_Function) -> int:
        return self.Terminal(Kind.BDD, value)

    # ────────────── operators ──────────────

    def Not(self, child: int) -> int:
        n = self._new_node(kind=Kind.NOT)
        self.T.add_left(n, child)
        return n

    def And(self, left: int, right: int) -> int:
        n = self._new_node(kind=Kind.AND)
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
