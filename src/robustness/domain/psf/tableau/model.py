from robustness.domain.types import _PSF_Type, _PSF_BDD_Type
from anytree import NodeMixin
from robustness.domain.logging import get_logger

logger = get_logger(__name__)

class TableauNode(NodeMixin):
    psf: _PSF_Type
    assignment: dict[str, int]

    def __init__(self, psf: _PSF_Type, parent: NodeMixin = None, assignment: dict[str, int] | None = None) -> None:
        logger.debug(f"Creating TableauNode with assignment: {assignment}")
        self.psf = psf
        self.parent = parent
        self.assignment = assignment

    def __str__(self) -> str:
        return f"{self.psf}[{self.assignment}]"

    def __repr__(self) -> str:
        return self.__str__()

    def is_bdd(self) -> bool:
        return isinstance(self.psf, _PSF_BDD_Type)

class TableauTree:
    root: TableauNode

    def __init__(self, root: TableauNode) -> None:
        logger.info("Creating new TableauTree")
        self.root = root

    def is_final(self) -> bool:
        return is_final(self.root)


def is_final(root: TableauNode) -> bool:
    logger.debug(f"Checking if tableau node is final: {len(root.children)} children")
    if len(root.children) == 0:
        return root.is_bdd()

    curr = True
    for child in root.children:
        curr &= is_final(child)
        if not curr:
            break

    return curr

def get_leaves(root: TableauNode) -> list[TableauNode]:
    logger.debug("Getting leaves from tableau node")
    if root.is_leaf():
        return [root]

    child_leaves = []
    for child in root.children:
        child_leaves += get_leaves(child)
    
    logger.debug(f"Found {len(child_leaves)} leaves")
    return child_leaves