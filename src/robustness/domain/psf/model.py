from __future__ import annotations

import abc
from typing import Generic, TypeVar

from anytree import NodeMixin

from robustness.domain.bdd import DD_Function


class PSF(abc.ABC, NodeMixin):
    _variables: set[str]
    _classes: set[str]

    def __init__(self, parent=None):
        self.parent = parent


T = TypeVar("T")


class Terminal(PSF, Generic[T]):
    __value: T

    def __init__(self, value: T, parent=None) -> None:
        super().__init__(parent)
        self.__value = value

    def __str__(self) -> str:
        return str(self.__value)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Terminal):
            return False

        return value.value == self.value

    value = property(lambda self: self.__value)


class Constant(Terminal[bool]):
    pass


class Variable(Terminal[str]):
    pass


class BDD(Terminal[DD_Function]):
    __count: dict[str, int]


class ClassNode(Terminal[str]):
    pass


class UnaryOperator(PSF, abc.ABC):
    _child: PSF

    def __init__(self, op_str: str, child: PSF, parent=None) -> None:
        super().__init__(parent)
        self._op_str = op_str
        self._child = child
        self._child.parent = self

    child = property(lambda self: self._child)

    def __str__(self) -> str:
        return f"{self._op_str}{self._child}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, UnaryOperator):
            return False

        if self._op_str != value._op_str:
            return False

        return value.child == self.child


class Not(UnaryOperator):

    def __init__(self, child: PSF, parent=None):
        super().__init__("~", child, parent)


class BinaryOperator(PSF, abc.ABC):
    _left_child: PSF
    _right_child: PSF

    def __init__(self, op_str: str, left_child: PSF, right_child: PSF, parent=None) -> None:
        super().__init__(parent)
        self._op_str = op_str
        self._left_child = left_child
        self._right_child = right_child

        self._left_child.parent = self
        self._right_child.parent = self

    left_child = property(lambda self: self._left_child)
    right_child = property(lambda self: self._right_child)

    def __str__(self) -> str:
        return f"{self.left_child} {self._op_str} {self.right_child}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, BinaryOperator):
            return False
        if self._op_str != value._op_str:
            return False

        return (
                value.left_child == self.left_child
                and value.right_child == self.right_child
        )


class And(BinaryOperator):

    def __init__(self, left_child: PSF, right_child: PSF, parent=None):
        super().__init__("&", left_child, right_child, parent)


def Or(left_child: PSF, right_child: PSF) -> PSF:
    """
    De Morgan rule.
        a | b = !(!a & !b)
    Args:
        left_child:
        right_child:

    Returns:

    """
    return Not(And(Not(left_child), Not(right_child)))
