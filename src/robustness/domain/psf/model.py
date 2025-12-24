from __future__ import annotations

import abc
from typing import Generic, TypeVar

import dd


class Formula(abc.ABC):
    pass


T = TypeVar("T")


class Terminal(Formula, Generic[T]):
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


class Not(Formula):
    __child: Formula

    def __init__(self, child: Formula) -> None:
        self.__child = child

    def __str__(self) -> str:
        return f"(! {self.__child})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Not):
            return False

        return value.child == self.child

    child = property(lambda self: self.__child)


class BinaryOperator(Formula, abc.ABC):
    __left_child: Formula
    __right_child: Formula

    def __init__(self, left_child: Formula, right_child: Formula) -> None:
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


class PSF:
    """
    PSF class represents a partially solved formula (PSF).

    Attributes:
        formula_str (str): A string representation of the formula.
        formula (Formula): An instance of the Formula class representing the formula.

    Methods:
        __init__(formula_str: str, formula: Formula) -> None:
            Initializes a new instance of the PSF class with the given formula string and formula.

        from_formula_str(cls, formula_str: str):
            Creates a PSF instance from a given formula string.
    """

    def __init__(self, formula_str: str, formula: Formula, assignment: dict[str, bool] | None = None) -> None:
        self.formula_str = formula_str
        self.formula = formula
        self.assignment = assignment

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, PSF):
            return False

        return value.formula == self.formula
