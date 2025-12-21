from __future__ import annotations
import abc
from pathlib import Path


class Formula(abc.ABC):
    pass


class Constant(Formula):
    __value: bool

    def __init__(self, value: bool) -> None:
        self.__value = value

    value = property(lambda self: self.__value)


class Variable(Formula):
    __name: str

    def __init__(self, name: str) -> None:
        self.__name = name

    name = property(lambda self: self.__name)


class Not(Formula):
    __child: Formula

    def __init__(self, child: Formula) -> None:
        self.__child = child

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
    pass


class Or(BinaryOperator):
    pass


def Or_DM(left_child: Formula, right_child: Formula):
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
        pass

    @classmethod
    def from_formula_str(cls, formula_str: str) -> PSF:
        return PSF(formula_str, parse_psf(formula_str))


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


"""
Partially Satisfiable Formula Grammar

PSF := True | False | VAR | NOT PSF | PSF AND PSF
NOT := not | Not | ! | ~
AND := and | And | && | &
VAR := \\w+
"""

from lark import Token, Transformer


grammar = """

start: dnf

?dnf: term
    | term OR dnf                   

?term: literal
     | literal AND term  

?literal: NOT VARIABLE
        | CONSTANT                  
        | VARIABLE              

AND.2: "and" | "And" | "AND" | "&&" | "&"
OR.2:  "or"  | "Or"  | "OR"  | "||" | "|"
NOT.2: "not" | "Not" | "NOT" | "!" | "~"

CONSTANT.2: "true" | "false"
VARIABLE: WORD

%import common.WORD
%import common.WS
%ignore WS
%ignore "("
%ignore ")"
"""


def ast_to_formula(lark_tree) -> Formula:
    if isinstance(lark_tree, Token):
        if lark_tree.type == "CONSTANT":
            return Constant(lark_tree.value == "true")
        if lark_tree.type == "VARIABLE":
            return Variable(lark_tree.value)
    else:
        if lark_tree.data == "start":
            return ast_to_formula(lark_tree.children[0])
        if lark_tree.data == "literal":
            child = lark_tree.children[0]
            if child.type == "NOT":
                child = ast_to_formula(lark_tree.children[1])
                return Not(child)
            return ast_to_formula(child)
        if lark_tree.data in {"term", "dnf"}:
            if len(lark_tree.children) == 0:
                return ast_to_formula(lark_tree.children[0])
            left = ast_to_formula(lark_tree.children[0])
            right = ast_to_formula(lark_tree.children[2])
            if lark_tree.data == "term":
                return And(left, right)
            else:
                return Or(left, right)

    raise TypeError(f"Unkown token {lark_tree}")


def parse_psf(formula_str: str) -> Formula:
    import lark as l

    parser = l.Lark(grammar=grammar, parser="lalr")

    lark_tree = parser.parse(formula_str)  # type: ignore
    return ast_to_formula(lark_tree.children[0])
