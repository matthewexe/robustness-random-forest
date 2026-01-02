from . import DD_Manager

from robustness.domain.types import _PSF_Type
from robustness.domain.utils.formula import get_variables, get_classes
from robustness.domain.utils.singleton import Singleton
from robustness.domain.logging import get_logger

logger = get_logger(__name__)


class BDDManager(Singleton):
    manager: DD_Manager
    _type: str

    def __init__(self, manager: DD_Manager):
        logger.debug("Initializing BDDManager singleton")
        self.manager = manager


def get_bdd_manager() -> DD_Manager:
    logger.debug("Getting BDD manager instance")
    return BDDManager(DD_Manager()).manager


def declare_vars(bdd: DD_Manager, formula: _PSF_Type):
    logger.info("Declaring BDD variables from formula")
    variables = get_variables(formula) | get_classes(formula)
    logger.debug(f"Declaring {len(variables)} variables in BDD manager")
    bdd.declare(*variables)
    logger.info(f"Successfully declared {len(variables)} variables")
    return bdd
