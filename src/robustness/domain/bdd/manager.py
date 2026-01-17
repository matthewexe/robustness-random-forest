from robustness.domain.logging import get_logger
from robustness.domain.types import _PSF_Type
from robustness.domain.utils.formula import get_variables, get_classes
from . import DD_Manager
from ..config import Config

logger = get_logger(__name__)
config = Config()

# class BDDManager(Singleton):
#     manager: DD_Manager
#     _type: str
#
#     def __init__(self, manager: DD_Manager):
#         logger.debug("Initializing BDDManager singleton")
#         self.manager = manager
#         if config.debug_mode:
#             self.manager.configure(reordering=False)


managers: list[DD_Manager] = []


def create_bdd_manager() -> DD_Manager:
    logger.debug("Getting BDD manager instance")
    manager = DD_Manager()
    if config.debug_mode:
        manager.configure(reordering=False)
        logger.debug("BDD manager configured with reordering disabled (debug mode)")
    else:
        logger.debug("BDD manager using default configuration(reordering=true)")
        manager.configure(reordering=True)

    managers.append(manager)

    return manager


def union_manager(*managers_to_union: DD_Manager) -> DD_Manager:
    logger.info("Creating a union of multiple BDD managers")
    new_manager = create_bdd_manager()
    all_vars = set()
    for man in managers_to_union:
        all_vars |= man.vars.keys()

    new_manager.declare(*all_vars)
    logger.info(f"Successfully created union BDD manager with {len(all_vars)} variables")
    return new_manager


def declare_vars(bdd: DD_Manager, formula: _PSF_Type):
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
