import dd
from dd.autoref import BDD

from robustness.domain.utils.singleton import Singleton


class BDDManager(Singleton):
    manager: BDD
    _type: str

    def __init__(self, manager: BDD):
        self.manager = manager


def get_bdd_manager() -> dd.autoref.BDD:
    return BDDManager(BDD()).manager
