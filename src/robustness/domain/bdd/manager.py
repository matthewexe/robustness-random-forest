from . import DD_Manager

from robustness.domain.types import _PSF_Type
from robustness.domain.utils.formula import get_variables, get_classes
from robustness.domain.utils.singleton import Singleton


class BDDManager(Singleton):
    manager: DD_Manager
    _type: str

    def __init__(self, manager: DD_Manager):
        self.manager = manager


def get_bdd_manager() -> DD_Manager:
    return BDDManager(DD_Manager()).manager


def declare_vars(bdd: DD_Manager, formula: _PSF_Type):
    variables = get_variables(formula) | get_classes(formula)
    bdd.declare(*variables)
    return bdd
