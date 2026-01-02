from __future__ import annotations

from typing import Iterable

from robustness.domain.config import Config
from robustness.domain.logging import get_logger
from robustness.domain.psf.model import PSF, VariableKind, ClassKind

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
    variables_list = list(variables)
    filtered = {var for var in variables_list if var.startswith(config.prefix_var)}
    logger.debug(f"Filtered {len(filtered)} variables out of {len(variables_list)} with prefix '{config.prefix_var}'")
    return filtered


def get_leaves(f: PSF) -> list[dict]:
    final_nodes = list()
    for n, attrs in f.nodes(data=True):
        if attrs['is_terminal']:
            final_nodes.append((n,attrs))
    return final_nodes


def get_variables(formula: PSF) -> set[str]:
    leaves = get_leaves(formula)
    filtered = filter(lambda x: x[1]['kind'] == VariableKind, leaves)
    return set(map(str, filtered))


def get_classes(formula: PSF) -> set[str]:
    leaves = get_leaves(formula)
    filtered = filter(lambda x: x[1]['kind'] == ClassKind, leaves)
    return set(map(str, filtered))
