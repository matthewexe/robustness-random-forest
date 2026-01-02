from __future__ import annotations

from robustness.domain.bdd.manager import get_bdd_manager
from robustness.domain.bdd.operations import max_occ_var
from robustness.domain.config import Config
from robustness.domain.psf.model import PSF, Terminal, Not, And, Variable, Constant, BDD, ClassNode, UnaryOperator, \
    BinaryOperator
from robustness.domain.psf.tableau.model import TableauTree, TableauNode
from robustness.domain.logging import get_logger

logger = get_logger(__name__)
bdd_manager = get_bdd_manager()


def simplify(f: PSF):
    logger.debug(f"Simplifying PSF formula of type {type(f).__name__}")
    if isinstance(f, Terminal):
        return f
    if isinstance(f, Not) and isinstance(f.child, Not):
        return simplify(f.child.child)
    if isinstance(f, Not):
        return Not(simplify(f.child))
    if isinstance(f, And):
        return And(simplify(f.left_child), simplify(f.right_child))

    raise TypeError(f"{type(f)} not recognized.")


def let(f: PSF, reduce: bool = False, **assignment) -> PSF:
    logger.debug(f"Applying let operation with {len(assignment)} assignments, reduce={reduce}")
    if isinstance(f, Terminal):
        if isinstance(f, Variable):
            return Constant(assignment[f.value]) if f.value in assignment else f
        if isinstance(f, BDD):
            global bdd_manager
            assign = {v: assignment[v] for v in bdd_manager.support(f.value) if v in assignment}
            return BDD(bdd_manager.let(assign, f.value))

        return f
    if isinstance(f, Not):
        sub_formula = let(f.child, reduce=reduce, **assignment)
        if isinstance(sub_formula, Constant) and reduce:
            return Constant(not sub_formula.value)

        return Not(sub_formula)
    if isinstance(f, And):
        left_formula = let(f.left_child, reduce=reduce, **assignment)
        right_formula = let(f.right_child, reduce=reduce, **assignment)

        if isinstance(left_formula, Constant) and left_formula.value == False and reduce:
            return Constant(False)
        if isinstance(right_formula, Constant) and right_formula.value == False and reduce:
            return Constant(False)
        if isinstance(left_formula, Constant) and isinstance(right_formula, Constant) and reduce:
            if left_formula.value == True and right_formula.value == True:
                return Constant(True)

        return And(let(f.left_child, reduce=reduce, **assignment), let(f.right_child, reduce=reduce, **assignment))

    raise TypeError(f"{type(f)} not recognized.")


def partial_reduce(f: PSF, diagram_size: int, **assignment) -> tuple[PSF, bool]:
    global bdd_manager
    logger.debug(f"Starting partial_reduce with diagram_size={diagram_size}, type={type(f).__name__}, {assignment=}")
    if assignment is None:
        assignment = {}
    if isinstance(f, ClassNode):
        class_label = f.value
        bdd = bdd_manager.add_expr(class_label)

        return BDD(bdd), True
    if isinstance(f, Variable):
        p = f.value
        bdd = bdd_manager.add_expr(p)
        if p in assignment:
            bdd = bdd_manager.let({p: assignment[p]}, bdd)
            return BDD(bdd), True

        return BDD(bdd), True

    if isinstance(f, BDD):
        current_bdd = f.value
        support = bdd_manager.support(current_bdd)
        assign = {v: assignment[v] for v in support if v in assignment}
        new_bdd = bdd_manager.let(assign, current_bdd)
        return BDD(new_bdd), new_bdd.dag_size <= diagram_size

    if isinstance(f, Not):
        node, outcome = partial_reduce(f.child, diagram_size, **assignment)
        if isinstance(node, BDD):
            return BDD(bdd_manager.apply('not', node.value)), outcome
        return Not(node), False

    if isinstance(f, And):
        node1, outcome1 = partial_reduce(f.left_child, diagram_size, **assignment)
        node2, outcome2 = partial_reduce(f.right_child, diagram_size, **assignment)

        if outcome1 and outcome2:
            new_bdd = bdd_manager.apply('and', node1.value, node2.value)
            return BDD(new_bdd), new_bdd.dag_size <= diagram_size

        return And(node1, node2), False

    raise TypeError(f"{type(f)} not recognized.")


def bdd_best_var(f: PSF) -> tuple[str, int]:
    logger.debug(f"Finding best variable in BDD for type {type(f).__name__}")
    if isinstance(f, Terminal):
        if isinstance(f, BDD):
            return max_occ_var(f.value)
        return "", 0
    if isinstance(f, UnaryOperator):
        return bdd_best_var(f.child)
    if isinstance(f, BinaryOperator):
        left_var, left_occurrences = bdd_best_var(f.left_child)
        right_var, right_occurrences = bdd_best_var(f.right_child)
        if left_occurrences >= right_occurrences:
            return left_var, left_occurrences

        return right_var, right_occurrences

    raise TypeError(f"{type(f)} not recognized.")


def tableau_method(f: PSF, tree: TableauNode | None = None, current_node: TableauNode | None = None) -> TableauTree:
    logger.debug(f"Applying tableau method on PSF type {type(f).__name__}")
    if not tree:
        logger.info("Creating new tableau tree")
        current_node = current_node or TableauNode(f)
        tree = TableauTree(current_node)
    if current_node.is_bdd():
        logger.debug("Current node is BDD, returning tree")
        return tree

    best_var = None
    best_var_occ = 0
    for leaf in tree.root.leaves:
        leaf_var, leaf_occ = bdd_best_var(leaf.psf)
        if best_var_occ < leaf_occ:
            best_var = leaf_var
            best_var_occ = leaf_occ
    
    logger.debug(f"Best variable found: {best_var} with {best_var_occ} occurrences")

    config_cls = Config()
    best_var_false = {best_var: False}
    logger.debug(f"Processing low branch with {best_var}=False")
    low_psf, _ = partial_reduce(current_node.psf, config_cls.diagram_size, **best_var_false)
    low_node = TableauNode(low_psf, current_node, best_var_false)
    tableau_method(low_psf, tree, low_node)

    best_var_true = {best_var: True}
    logger.debug(f"Processing high branch with {best_var}=True")
    high_psf, _ = partial_reduce(current_node.psf, config_cls.diagram_size, **best_var_true)
    high_node = TableauNode(high_psf, current_node, best_var_true)
    tableau_method(high_psf, tree, high_node)
    
    logger.debug("Tableau method completed for current node")
    return tree
