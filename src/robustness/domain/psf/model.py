from __future__ import annotations

import abc
from typing import Generic, TypeVar

import dd


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


class BDD(Terminal[dd.autoref.Function]):
    pass

class ClassNode(Terminal[str]):
    pass

class UnaryOperator(PSF, abc.ABC):
    __child: PSF

    def __init__(self, child: PSF) -> None:
        self.__child = child

    child = property(lambda self: self.__child)

class Not(UnaryOperator):

    def __str__(self) -> str:
        return f"(! {self.__child})"

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


class Or(BinaryOperator):

    def __str__(self) -> str:
        return f"({self.left_child}) || {self.right_child}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Or):
            return False

        return (
                value.left_child == self.left_child
                and value.right_child == self.right_child
        )
