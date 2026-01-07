from . import DD_Function
from .manager import get_bdd_manager
from ..utils.formula import filter_variables
from robustness.domain.logging import get_logger

logger = get_logger(__name__)

manager = get_bdd_manager()


def count_vars(f: DD_Function, *variables) -> int:
    """
    Assuming DAG

    Args:
        f:
        *variables:

    Returns:

    """
    count = sum([1 for node in iter_nodes(f) if node.var in variables])
    logger.debug(f"Counted {count} occurrences of {", ".join(variables)} in BDD")
    return count

def iter_nodes(root):
    visited = set()
    stack = [root]

    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)

        yield u

        if u.var is not None:  # non-terminal
            stack.append(u.low)
            stack.append(u.high)

def max_occ_var(f: DD_Function) -> tuple[str, int]:
    logger.debug("Finding variable with maximum occurrences in BDD")
    count = {v:count_vars(f, v) for v in manager.support(f)}
    variables = filter_variables(count.keys())
    if len(variables) == 0:
        logger.debug("No variables found in BDD")
        return "", -1

    filtered_count = {v:count[v] for v in variables}
    best_var = max(filtered_count, key=filtered_count.get)
    logger.debug(f"Best variable: {best_var} with {filtered_count[best_var]} occurrences")
    return best_var, filtered_count[best_var]