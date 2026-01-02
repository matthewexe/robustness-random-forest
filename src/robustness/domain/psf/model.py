from __future__ import annotations

from typing import TypeAlias, TypedDict, Literal

import networkx as nx

from robustness.domain.bdd import DD_Function

PSF: TypeAlias = nx.DiGraph
_KindType: TypeAlias = Literal["Constant", "Variable", "Class", "BDD", "Not", "And"]
ConstantKind = "Constant"
VariableKind = "Variable"
ClassKind = "Class"
BDDKind = "BDD"
NotKind = "Not"
AndKind = "And"

def int_generator(start=0):
    i = start
    while True:
        yield i
        i+=1

class Builder:
    _digraph: PSF

    def __init__(self):
        self._digraph = nx.DiGraph()
        self.__id_gen = int_generator()

    def __next_id(self) -> int:
        return next(self.__id_gen)

    def Terminal(self, kind: str, value: object) -> int:
        node_id = self.__next_id()
        self._digraph.add_node(node_id, kind=kind, value=value, is_terminal=True)
        return node_id

    def Constant(self,value: bool) -> int:
        return self.Terminal(ConstantKind, value)

    def Variable(self, value: str) -> int:
        return self.Terminal(VariableKind, value)

    def Class(self, value: str) -> int:
        return self.Terminal(ClassKind, value)

    def BDD(self, value: DD_Function) -> int:
        return self.Terminal(BDDKind, value)

    def Operator(self, kind: str, children: list[int]) -> int:
        node_id = self.__next_id()
        self._digraph.add_node(node_id, kind=kind, is_terminal=False)
        for child in children:
            self._digraph.add_edge(node_id, child)

        return node_id

    def Not(self, child: int) -> int:
        return self.Operator(NotKind, children=[child])

    def And(self, left_child: int, right_child: int) -> int:
        return self.Operator(AndKind, children=[left_child, right_child])

    def Or(self, left_child: int, right_child : int):
        """
        De morgan rule
        Args:
            left_child:
            right_child:

        Returns:

        """

        return self.Not(self.And(self.Not(left_child), self.Not(right_child)))

    def build(self) -> PSF:
        return self._digraph

# T = TypeVar("T")
#
#
# class Terminal(PSF, Generic[T]):
#     _value: T
#
#     def __init__(self, value: T, parent=None) -> None:
#         super().__init__(parent)
#         self._value = value
#
#     def __str__(self) -> str:
#         return str(self._value)
#
#     def __repr__(self) -> str:
#         return self.__str__()
#
#     def __hash__(self):
#         return hash(self._value)
#
#     def __eq__(self, value: object) -> bool:
#         if not isinstance(value, Terminal):
#             return False
#
#         return value._value == self._value
#
#     value = property(lambda self: self._value)
#
#
# class Constant(Terminal[bool]):
#     pass
#
#
# class Variable(Terminal[str]):
#     pass
#
#
# class BDD(Terminal[DD_Function]):
#     __count: dict[str, int]
#
#
# class ClassNode(Terminal[str]):
#     pass
#
#
# class UnaryOperator(PSF, abc.ABC):
#     _child: PSF
#
#     def __init__(self, op_str: str, child: PSF, parent=None) -> None:
#         super().__init__(parent)
#         self._op_str = op_str
#         self._child = child
#         self._child.parent = self
#
#     child = property(lambda self: self._child)
#
#     def __str__(self) -> str:
#         return f"{self._op_str}{self._child}"
#
#     def __repr__(self) -> str:
#         return self.__str__()
#
#     def __hash__(self):
#         return hash((self._op_str, self.child))
#
#     def __eq__(self, value: object) -> bool:
#         if not isinstance(value, UnaryOperator):
#             return False
#
#         if self._op_str != value._op_str:
#             return False
#
#         return value.child == self.child
#
#
# class Not(UnaryOperator):
#
#     def __init__(self, child: PSF, parent=None):
#         super().__init__("~", child, parent)
#
#
# class BinaryOperator(PSF, abc.ABC):
#     _left_child: PSF
#     _right_child: PSF
#
#     def __init__(self, op_str: str, left_child: PSF, right_child: PSF, parent=None) -> None:
#         super().__init__(parent)
#         self._op_str = op_str
#         self._left_child = left_child
#         self._right_child = right_child
#
#         self._left_child.parent = self
#         self._right_child.parent = self
#
#     left_child = property(lambda self: self._left_child)
#     right_child = property(lambda self: self._right_child)
#
#     def __str__(self) -> str:
#         return f"{self.left_child} {self._op_str} {self.right_child}"
#
#     def __repr__(self) -> str:
#         return self.__str__()
#
#     def __hash__(self):
#         return hash((self._op_str, self._left_child, self.right_child))
#
#     def __eq__(self, value: object) -> bool:
#         if not isinstance(value, BinaryOperator):
#             return False
#         if self._op_str != value._op_str:
#             return False
#
#         return (
#                 value.left_child == self.left_child
#                 and value.right_child == self.right_child
#         )
#
#
# class And(BinaryOperator):
#
#     def __init__(self, left_child: PSF, right_child: PSF, parent=None):
#         super().__init__("&", left_child, right_child, parent)
#
#
# def Or(left_child: PSF, right_child: PSF) -> PSF:
#     """
#     De Morgan rule.
#         a | b = !(!a & !b)
#     Args:
#         left_child:
#         right_child:
#
#     Returns:
#
#     """
#     return Not(And(Not(left_child), Not(right_child)))
