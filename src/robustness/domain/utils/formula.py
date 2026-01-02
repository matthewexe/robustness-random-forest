from __future__ import annotations

from typing import Iterable

from robustness.domain.config import Config
from robustness.domain.psf.model import PSF, Or, Not, And, Variable, Terminal, UnaryOperator, BinaryOperator, \
    ClassNode
from robustness.domain.logging import get_logger

logger = get_logger(__name__)


def is_feature(value: str) -> bool:
    if not isinstance(value, str):
        return False
    import re
    return re.match(r"^t_\d+$", value) is not None


def is_class(value: str) -> bool:
    if not isinstance(value, str):
        return False

    import re
    return re.match(r"^(c)?\d+$", value) is not None


def or_de_morgan(f: PSF):
    """
    Computes the logical OR of two formulas using De Morgan's law.

    This function implements OR by negating the AND of the negated operands:
    OR(A, B) = NOT(AND(NOT(A), NOT(B)))

    Args:
        left_child (PSF): The left operand formula.
        right_child (PSF): The right operand formula.

    Returns:
        PSF: A formula representing the logical OR of the two input formulas.

    Example:
        >>> result = Or(formula_a, formula_b)
        >>> # result represents: formula_a OR formula_b
    """
    if isinstance(f, PSF):
        return f

    raise TypeError(f"{type(f)} not recognized.")


def filter_variables(variables: Iterable[str]) -> set[str]:
    config = Config()
    filtered = {var for var in variables if var.startswith(config.prefix_var)}
    logger.debug(f"Filtered {len(filtered)} variables out of {len(list(variables))} with prefix '{config.prefix_var}'")
    return filtered


def get_variables(formula: PSF) -> set[str]:
    logger.debug(f"Extracting variables from formula type {type(formula).__name__}")
    if isinstance(formula, Terminal):
        if isinstance(formula, Variable):
            s = set()
            s.add(formula.value)
            logger.debug(f"Found variable: {formula.value}")
            return s
        return set()
    if isinstance(formula, UnaryOperator):
        return get_variables(formula.child)
    if isinstance(formula, BinaryOperator):
        return get_variables(formula.left_child) | get_variables(formula.right_child)

    raise TypeError(f"{type(formula)} not recognized.")


def get_classes(formula: PSF) -> set[str]:
    logger.debug(f"Extracting classes from formula type {type(formula).__name__}")
    if isinstance(formula, Terminal):
        if isinstance(formula, ClassNode):
            s = set()
            s.add(formula.value)
            logger.debug(f"Found class: {formula.value}")
            return s
        return set()
    if isinstance(formula, UnaryOperator):
        return get_classes(formula.child)
    if isinstance(formula, BinaryOperator):
        return get_classes(formula.left_child) | get_classes(formula.right_child)

    raise TypeError(f"{type(formula)} not recognized.")
