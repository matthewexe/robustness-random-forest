from __future__ import annotations

import abc
from typing import Generic, TypeVar

from robustness.domain.bdd import DD_Function


class PSF(abc.ABC):
    pass


T = TypeVar("T")


class Terminal(PSF, Generic[T]):
    __value: T

    def __init__(self, value: T) -> None:
        self.__value = value

    def __str__(self) -> str:
        return str(self.__value)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Constant):
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

    def __init__(self, child: PSF) -> None:
        self._child = child

    child = property(lambda self: self._child)


class Not(UnaryOperator):

    def __str__(self) -> str:
        return f"(! {self._child})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Not):
            return False

        return value.child == self.child


class BinaryOperator(PSF, abc.ABC):
    __left_child: PSF
    __right_child: PSF

    def __init__(self, left_child: PSF, right_child: PSF) -> None:
        self.__left_child = left_child
        self.__right_child = right_child

    left_child = property(lambda self: self.__left_child)
    right_child = property(lambda self: self.__right_child)


class And(BinaryOperator):

    def __str__(self) -> str:
        return f"{self.left_child} && {self.right_child}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, And):
            return False

        return (
                value.left_child == self.left_child
                and value.right_child == self.right_child
        )

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

# class Or(BinaryOperator):
#
#     def __str__(self) -> str:
#         return f"({self.left_child}) || {self.right_child}"
#
#     def __repr__(self) -> str:
#         return self.__str__()
#
#     def __eq__(self, value: object) -> bool:
#         if not isinstance(value, Or):
#             return False
#
#         return (
#                 value.left_child == self.left_child
#                 and value.right_child == self.right_child
#         )
