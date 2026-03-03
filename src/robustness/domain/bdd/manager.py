from dd.autoref import BDD

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

__manager: DD_Manager | None = None


def get_id_of_manager(manager: DD_Manager) -> int | None:
    return id(manager)


def get_bdd_manager() -> DD_Manager:
    """
    Singleton pattern for BDD manager instance. If an instance already exists, it returns the existing one.

    Returns: A tuple containing the BDD manager instance and its unique ID.

    """
    global __manager

    logger.debug("Getting BDD manager instance")
    if __manager is None:
        __manager = DD_Manager()
        __manager.configure(reordering=None)
        logger.debug("BDD manager created successfully")
    else:
        logger.debug("BDD manager instance already exists, returning existing instance")

    return __manager

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
    logger.info("Cleaning up BDD manager")
    if __manager is not None:
        __manager.collect_garbage()
    logger.info("BDD manager cleaned up successfully")
