from typing import Iterator

from robustness.domain.logging import get_logger
from robustness.domain.random_forest import Endpoints, Sample
from robustness.domain.utils.formula import filter_features, is_class
from . import DD_Function, DD_Manager

logger = get_logger(__name__)


def count_vars(bdd: DD_Function, *variables) -> int:
    """
    It counts the number of occurrences of the given variables in bdd.

    Args:
        bdd: bdd to analyze
        *variables: variables to count in the bdd

    Returns: the count of occurrences of the given variables in the bdd

    """
    count = sum([1 for node in iter_bdd_nodes(bdd) if node.var in variables])
    logger.debug(f"Counted {count} occurrences of {", ".join(variables)} in BDD")
    return count


def iter_bdd_nodes(bdd: DD_Function) -> Iterator[DD_Function]:
    """
    It returns an iterator that visit BDD nodes in dfs order.

    Args:
        bdd: bdd node to start the iteration from

    Returns: an iterator that yields BDD nodes in dfs order

    """
    visited = set()
    stack = [bdd]

    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)

        yield u

        if u.var is not None:  # non-terminal
            stack.append(u.low)
            stack.append(u.high)


def get_vars_counting(bdd: DD_Function) -> dict[str, int]:
    """
    Given a BDD, it returns a dictionary that maps each variable with its number of occurrences in the BDD.

    Args:
        bdd: bdd to analyze

    Returns: a dictionary mapping each variable to its count of occurrences in the BDD

    """
    return {v: count_vars(bdd, v) for v in bdd.support}


def features_counting(bdd: DD_Function) -> dict[str, int]:
    """
    Given a BDD, it returns a dictionary that maps each feature variable with its number of occurrences in the BDD.
    It filters out class variables from vars_counting.

    Args:
        bdd: bdd to analyze

    Returns: a dictionary mapping each feature variable to its count of occurrences in the BDD

    """
    variables = filter_features(bdd.support)
    count = get_vars_counting(bdd)
    return {v: count[v] for v in variables}


def max_occ_var(bdd: DD_Function) -> tuple[str, int]:
    """
    Given a BDD, it returns the variable with the maximum number of occurrences in the BDD and its count.

    Args:
        bdd: bdd to analyze

    Returns: max variable and its count of occurrences in the BDD

    """
    logger.debug("Finding variable with maximum occurrences in BDD")
    filtered_count = features_counting(bdd)

    if len(filtered_count) == 0:
        logger.debug("No variables found in BDD")
        return "", -1

    best_var = max(filtered_count, key=filtered_count.get)
    logger.debug(f"Best variable: {best_var} with {filtered_count[best_var]} occurrences")
    return best_var, filtered_count[best_var]


def test_sample(sample: Sample, manager: DD_Manager, f: DD_Function, endpoints: Endpoints) -> str:
    """
    Given a sample, a BDD manager, a BDD function and the endpoints universe,
    it returns a string of bits that represents tha path from BDD root to the leaf of the BDD that corresponds to the sample.
    The path is encoded as a string of bits, where "0" represents a low edge and "1" represents a high edge.

    Args:
        sample: the sample to test
        manager: the BDD manager
        f: the BDD function
        endpoints: the endpoints universe

    Returns: path encoding as string of bits

    """
    if f in {manager.false, manager.true}:
        return ""

    if is_class(f.var):
        if sample.predicted_label == f.var[1:]:
            return test_sample(sample, manager, f.high, endpoints) + "1"
        else:
            return test_sample(sample, manager, f.low, endpoints) + "0"

    if sample.features[f.var] <= endpoints[f.var]:
        return test_sample(sample, manager, f.low, endpoints) + "0"
    else:
        return test_sample(sample, manager, f.high, endpoints) + "1"


def write_bdd(manager: DD_Manager, value: DD_Function):
    """Write BDD to SVG file."""
    try:
        import os

        filename = f"{int(value)}.dot"
        manager.dump(os.path.join("logs/bdds", filename), roots=[value])
        logger.debug(f"BDD written to DOT file: {filename}")
    except Exception as e:
        logger.exception("Failed to write BDD to DOT file")
        raise e
