from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

from robustness.domain.config import Config
from robustness.domain.logging import get_logger

if TYPE_CHECKING:
    from robustness.domain.psf.model import PSF, Kind, is_terminal

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

def get_class_label(value: str) -> str:
    if not is_class(value):
        return value

    return value[1:]

def filter_variables(variables: Iterable[str]) -> set[str]:
    config = Config()
    variables_list = list(variables)
    filtered = {var for var in variables_list if var.startswith(config.prefix_var)}
    logger.debug(f"Filtered {len(filtered)} variables out of {len(variables_list)} with prefix '{config.prefix_var}'")
    return filtered


def get_leaves(f: PSF) -> list[dict]:
    final_nodes = list()
    for n, attrs in f.nodes(data=True):
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
