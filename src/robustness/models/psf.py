from __future__ import annotations
import abc


class Formula(abc.ABC):
    pass


class Constant(Formula):
    __value: bool

    def __init__(self, value: bool) -> None:
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


class Variable(Formula):
    __name: str

    def __init__(self, name: str) -> None:
        self.__name = name

    def __str__(self) -> str:
        return str(self.__name)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Variable):
            return False

        return value.name == self.name

    name = property(lambda self: self.__name)


class Not(Formula):
    __child: Formula

    def __init__(self, child: Formula) -> None:
        self.__child = child

    def __str__(self) -> str:
        return f"(not {self.__child})"

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
        return f"{self.left_child} and {self.right_child}"

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
        return f"({self.left_child}) or {self.right_child}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Or):
            return False

        return (
            value.left_child == self.left_child
            and value.right_child == self.right_child
        )


def Or_DeMorgan(left_child: Formula, right_child: Formula):
    """
    Computes the logical OR of two formulas using De Morgan's law.

    This function implements OR by negating the AND of the negated operands:
    OR(A, B) = NOT(AND(NOT(A), NOT(B)))

    Args:
        left_child (Formula): The left operand formula.
        right_child (Formula): The right operand formula.

    Returns:
        Formula: A formula representing the logical OR of the two input formulas.

    Example:
        >>> result = Or(formula_a, formula_b)
        >>> # result represents: formula_a OR formula_b
    """
    return Not(And(Not(left_child), Not(right_child)))


class PSF:
    """
    PSF class represents a partially satisfiable formula (PSF).

    Attributes:
        formula_str (str): A string representation of the formula.
        formula (Formula): An instance of the Formula class representing the formula.

    Methods:
        __init__(formula_str: str, formula: Formula) -> None:
            Initializes a new instance of the PSF class with the given formula string and formula.

        from_formula_str(cls, formula_str: str):
            Creates a PSF instance from a given formula string.
    """

    def __init__(self, formula_str: str, formula: Formula) -> None:
        self.formula_str = formula_str
        self.formula = formula

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, PSF):
            return False

        return value.formula == self.formula


def simplify(f: Formula):
    if isinstance(f, Constant | Variable):
        return f
    if isinstance(f, Not) and isinstance(f.child, Not):
        return f.child.child
    if isinstance(f, Not):
        return Not(simplify(f.child))
    if isinstance(f, And):
        return And(simplify(f.left_child), simplify(f.right_child))
    if isinstance(f, Or):
        return Or(simplify(f.left_child), simplify(f.right_child))

    raise TypeError(f"{type(f)} not recognized.")
