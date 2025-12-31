from . import DD_Function
from .manager import get_bdd_manager
from ..utils.formula import filter_variables

manager = get_bdd_manager()


def count_vars(f: DD_Function, *variables) -> int:
    """
    Assuming DAG

    Args:
        f:
        *variables:

    Returns:

    """
    return sum([1 for node in iter_nodes(f) if node.var in variables])

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
    count = {v:count_vars(f, v) for v in manager.support(f)}
    variables = filter_variables(count.keys())
    if len(variables) == 0:
        return "", -1

    filtered_count = {v:count[v] for v in variables}
    best_var = max(filtered_count, key=filtered_count.get)
    return best_var, filtered_count[best_var]