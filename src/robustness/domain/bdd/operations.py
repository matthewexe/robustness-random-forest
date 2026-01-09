from robustness.domain.logging import get_logger
from robustness.domain.random_forest import Endpoints, Sample
from robustness.domain.utils.formula import filter_variables, is_class
from . import DD_Function
from .manager import get_bdd_manager

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


def vars_counting(bdd: DD_Function) -> dict[str, int]:
    return {v: count_vars(bdd, v) for v in bdd.support}


def features_counting(bdd: DD_Function) -> dict[str, int]:
    variables = filter_variables(bdd.support)
    count = vars_counting(bdd)
    return {v: count[v] for v in variables}


def max_occ_var(f: DD_Function) -> tuple[str, int]:
    logger.debug("Finding variable with maximum occurrences in BDD")
    count = {v: count_vars(f, v) for v in manager.support(f)}
    variables = filter_variables(count.keys())
    if len(variables) == 0:
        logger.debug("No variables found in BDD")
        return "", -1

    filtered_count = {v: count[v] for v in variables}
    best_var = max(filtered_count, key=filtered_count.get)
    logger.debug(f"Best variable: {best_var} with {filtered_count[best_var]} occurrences")
    return best_var, filtered_count[best_var]


def path_of(sample: Sample, f: DD_Function, endpoints: Endpoints) -> str:
    if f in {manager.false, manager.true}:
        return ""

    if is_class(f.var):
        if sample.predicted_label == f.var[1:]:
            return path_of(sample, f.high, endpoints) + "1"
        else:
            return path_of(sample, f.low, endpoints) + "0"

    if sample.features[f.var] <= endpoints[f.var]:
        return path_of(sample, f.low, endpoints) + "0"
    else:
        return path_of(sample, f.high, endpoints) + "1"
