from robustness.domain.logging import get_logger
from robustness.domain.random_forest import Endpoints, Sample
from robustness.domain.utils.formula import filter_variables, is_class
from . import DD_Function, DD_Manager
from .manager import create_bdd_manager, union_manager, get_id_of_manager

logger = get_logger(__name__)


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


def iter_nodes(root: DD_Function):
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
    count = {v: count_vars(f, v) for v in f.support}
    variables = filter_variables(count.keys())
    if len(variables) == 0:
        logger.debug("No variables found in BDD")
        return "", -1

    filtered_count = {v: count[v] for v in variables}
    best_var = max(filtered_count, key=filtered_count.get)
    logger.debug(f"Best variable: {best_var} with {filtered_count[best_var]} occurrences")
    return best_var, filtered_count[best_var]


def path_of(sample: Sample, manager: DD_Manager, f: DD_Function, endpoints: Endpoints) -> str:
    if f in {manager.false, manager.true}:
        return ""

    if is_class(f.var):
        if sample.predicted_label == f.var[1:]:
            return path_of(sample, manager, f.high, endpoints) + "1"
        else:
            return path_of(sample, manager, f.low, endpoints) + "0"

    if sample.features[f.var] <= endpoints[f.var]:
        return path_of(sample, manager, f.low, endpoints) + "0"
    else:
        return path_of(sample, manager,f.high, endpoints) + "1"


def copy_from_to(manager_from: DD_Manager, value: DD_Function, manager_to: DD_Manager) -> DD_Function:
    """Copy BDD function from one manager to another."""
    try:
        expr = value.to_expr()
        new_value = manager_to.add_expr(expr)
        return new_value
    except Exception as e:
        logger.exception("Failed to copy BDD function between managers")
        raise e


def Not(manager: DD_Manager, value: DD_Function) -> tuple[DD_Manager, DD_Function, int]:
    """Logical NOT operation on BDD function."""
    try:
        if value == manager.true:
            return manager, manager.false, get_id_of_manager(manager)
        elif value == manager.false:
            return manager, manager.true, get_id_of_manager(manager)
        copy_manager, copy_manager_id = union_manager(manager)
        new_value = copy_from_to(manager, value, copy_manager)
        not_new_value = copy_manager.apply("not", new_value)
        return copy_manager, not_new_value, copy_manager_id
    except Exception as e:
        logger.exception("Failed to compute NOT for BDD")
        raise e


def And(manager_left: DD_Manager, left: DD_Function, manager_right: DD_Manager, right: DD_Function) -> tuple[
    DD_Manager, DD_Function, int]:
    """Logical AND operation on two BDD functions."""
    try:
        if left == manager_left.false or right == manager_right.false:
            manager, manager_id = create_bdd_manager()
            return manager, manager.false, manager_id
        elif left == manager_left.true:
            copy_manager, copy_manager_id = union_manager(manager_right)
            new_right = copy_from_to(manager_right, right, copy_manager)
            return copy_manager, new_right, copy_manager_id
        elif right == manager_right.true:
            copy_manager, copy_manager_id = union_manager(manager_left)
            new_left = copy_from_to(manager_left, left, copy_manager)
            return copy_manager, new_left, copy_manager_id

        copy_manager, copy_manager_id = union_manager(manager_left, manager_right)

        new_left = copy_from_to(manager_left, left, copy_manager)
        new_right = copy_from_to(manager_right, right, copy_manager)
        and_new_value = copy_manager.apply("and", new_left, new_right)
        return copy_manager, and_new_value, copy_manager_id
    except Exception as e:
        logger.exception("Failed to compute AND for BDDs")
        raise e


def write_bdd(manager: DD_Manager, value: DD_Function, id_manager: int | None = None):
    """Write BDD to DOT file."""
    try:
        import os
        if not id_manager:
            id_manager = get_id_of_manager(manager)

        filename = f"{id_manager}_{int(value)}.dot"
        manager.dump(os.path.join("logs/bdds", filename), roots=[value])
        logger.info(f"BDD written to DOT file: {filename}")
    except Exception as e:
        logger.exception("Failed to write BDD to DOT file")
        raise e
