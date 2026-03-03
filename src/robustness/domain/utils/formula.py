from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

from robustness.domain.config import Config
from robustness.domain.logging import get_logger
from robustness.domain.psf.model import Kind, is_terminal

if TYPE_CHECKING:
    from robustness.domain.psf.model import PSF

logger = get_logger(__name__)


def is_feature(variable_name: str) -> bool:
    """
    Check if variable name starts with config.prefix_var and is followed by digits.
    (Example with prefix_var="t_": t_0, t_1, t_2, ...)

    Args:
        variable_name: the variable name to check

    Returns: True if the variable name is a feature, False otherwise

    """
    if not isinstance(variable_name, str):
        return False
    import re
    return re.match(r"^t_\d+$", variable_name) is not None


def is_class(variable_name: str) -> bool:
    """
        Check if variable name starts with config.prefix_var and is followed by digits.
        (Example with prefix_var="t_": t_0, t_1, t_2, ...)

        Args:
            variable_name: the variable name to check

        Returns: True if the variable name is a feature, False otherwise

        """
    if not isinstance(variable_name, str):
        return False

    import re
    return re.match(r"^(c)?(_|-)?\d+$", variable_name) is not None

def get_class_label(variable_name: str) -> str:
    """
    Remove class prefix from the variable name and return the class label.

    Args:
        variable_name: the variable name to extract the class label from

    Returns: the class label if the variable name is a class, otherwise returns the original variable name

    """
    if not is_class(variable_name):
        return variable_name

    return variable_name[1:]

def filter_features(variables: Iterable[str]) -> set[str]:
    config = Config()
    variables_list = list(variables)
    filtered = {var for var in variables_list if is_feature(var)}
    logger.debug(f"Filtered {len(filtered)} variables out of {len(variables_list)} with prefix '{config.prefix_var}'")
    return filtered


def get_leaves(psf: PSF) -> list[tuple[int, dict]]:
    """
    Used to get leaves node id and its attributes.

    Args:
        psf: PSF formula to analyze

    Returns: a list of tuples, where each tuple contains the node id and its attributes for all terminal nodes in the PSF formula.

    """
    final_nodes = list()
    for n, attrs in psf.nodes(data=True):
        if is_terminal(attrs['kind']):
            final_nodes.append((n,attrs))
    return final_nodes


def get_variables(formula: PSF) -> set[str]:
    leaves = get_leaves(formula)
    filtered = filter(lambda x: x[1]['kind'] == Kind.VARIABLE, leaves)
    return set(map(lambda item: item[1]['value'], filtered))


def get_classes(formula: PSF) -> set[str]:
    leaves = get_leaves(formula)
    filtered = filter(lambda x: x[1]['kind'] == Kind.CLASS, leaves)
    return set(map(lambda item: item[1]['value'], filtered))
