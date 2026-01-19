from robustness.domain.logging import get_logger
from robustness.domain.utils.formula import get_variables, get_classes
from . import DD_Manager
from ..config import Config

logger = get_logger(__name__)
config = Config()

managers: list[DD_Manager] = []


def int_generator():
    n = 0
    while True:
        yield n
        n += 1


id_gen = int_generator()

memo = {}


def get_id_of_manager(manager: DD_Manager) -> int | None:
    return memo.get(id(manager), None)


def create_bdd_manager() -> tuple[DD_Manager, int]:
    logger.debug("Getting BDD manager instance")
    manager = DD_Manager()
    if config.debug_mode:
        manager.configure(reordering=None)
        logger.debug("BDD manager configured with reordering disabled (debug mode)")
    else:
        logger.debug("BDD manager using default configuration(reordering=true)")
        manager.configure(reordering=True)

    managers.append(manager)
    manager_id = next(id_gen)
    memo[id(manager)] = manager_id

    return manager, manager_id


def union_manager(*managers_to_union: DD_Manager) -> tuple[DD_Manager, int]:
    logger.info("Creating a union of multiple BDD managers")
    new_manager, new_manager_id = create_bdd_manager()
    all_vars = set()
    for man in managers_to_union:
        all_vars |= man.vars.keys()

    if config.debug_mode:
        _features = [v for v in all_vars if v.startswith(config.prefix_var)]
        classes = list(all_vars - set(_features))
        all_vars = list(sorted(_features)) + list(sorted(classes))
        logger.debug(f"Sorted variables for union: {all_vars}")

    new_manager.declare(*all_vars)
    logger.info(f"Successfully created union BDD manager with {len(all_vars)} variables")
    return new_manager, new_manager_id


def declare_vars(bdd: DD_Manager, formula: "robustness.domain.types._PSF_Type"):
    logger.info("Declaring BDD variables from formula")
    variables = get_variables(formula)
    classes = get_classes(formula)
    if config.debug_mode:
        all_variables = list(sorted(variables)) + list(sorted(classes))
        logger.debug(f"Variables to declare: {sorted(variables)}")
        logger.debug(f"Classes to declare: {sorted(classes)}")
    else:
        all_variables = variables | classes

    logger.debug(f"Declaring {len(all_variables)} variables in BDD manager: {", ".join(all_variables)}")
    bdd.declare(*variables, *classes)
    logger.info(f"Successfully declared {len(all_variables)} variables")
    return bdd


def cleanup_bdd_manager():
    logger.info("Cleaning up BDD managers")
    for man in managers:
        man.collect_garbage()

    logger.info("BDD managers cleaned up successfully")
