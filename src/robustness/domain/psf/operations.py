from __future__ import annotations

from robustness.domain.bdd import get_bdd_manager
from robustness.domain.psf.model import PSF, Terminal, Not, And, Or, Variable, Constant, BDD, ClassNode

bdd_manager = get_bdd_manager()


def simplify(f: PSF):
    if isinstance(f, Terminal):
        return f
    if isinstance(f, Not) and isinstance(f.child, Not):
        return simplify(f.child.child)
    if isinstance(f, Not):
        return Not(simplify(f.child))
    if isinstance(f, And):
        return And(simplify(f.left_child), simplify(f.right_child))
    if isinstance(f, Or):
        return Or(simplify(f.left_child), simplify(f.right_child))

    raise TypeError(f"{type(f)} not recognized.")


def let(f: PSF, reduce: bool = False, **assignment) -> PSF:
    if isinstance(f, Terminal):
        if isinstance(f, Variable):
            return Constant(assignment[f.value]) if f.value in assignment else f
        if isinstance(f, BDD):
            global bdd_manager
            assign = {v: assignment[v] for v in f.value.vars if v in assignment}
            return BDD(bdd_manager.let(assign, f.value))

        return f
    if isinstance(f, Not):
        sub_formula = let(f, reduce=reduce, **assignment)
        if isinstance(sub_formula, Constant) and reduce:
            return Constant(not sub_formula.value)

        return Not(sub_formula)
    if isinstance(f, And):
        left_formula = let(f, reduce=reduce, **assignment)
        right_formula = let(f, reduce=reduce, **assignment)

        if isinstance(left_formula, Constant) and left_formula.value == False and reduce:
            return Constant(False)
        if isinstance(right_formula, Constant) and right_formula.value == False and reduce:
            return Constant(False)
        if isinstance(left_formula, Constant) and isinstance(right_formula, Constant) and reduce:
            if left_formula.value == True and right_formula.value == True:
                return Constant(True)

        return And(let(f.left_child, reduce=reduce, **assignment), let(f.right_child, reduce=reduce, **assignment))
    if isinstance(f, Or):
        left_formula = let(f, reduce=reduce, **assignment)
        right_formula = let(f, reduce=reduce, **assignment)

        if isinstance(left_formula, Constant) and left_formula.value == True and reduce:
            return Constant(True)
        if isinstance(right_formula, Constant) and right_formula.value == True and reduce:
            return Constant(True)
        if isinstance(left_formula, Constant) and isinstance(right_formula, Constant) and reduce:
            if left_formula.value == False and right_formula.value == False:
                return Constant(False)

        return Or(let(f.left_child, reduce=reduce, **assignment), let(f.right_child, reduce=reduce, **assignment))

    raise TypeError(f"{type(f)} not recognized.")


def partial_reduce(f: PSF, diagram_size: int, assignment: dict) -> tuple[PSF, bool]:
    global bdd_manager
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
        return BDD(new_bdd), bdd_manager.count(new_bdd) > diagram_size

    if isinstance(f, Not):
        node, outcome = partial_reduce(f.child, diagram_size, assignment)
        if isinstance(node, BDD):
            return BDD(bdd_manager.apply('not', node.value)), outcome
        return Not(node), False

    if isinstance(f, And):
        node1, outcome1 = partial_reduce(f.left_child, diagram_size, assignment)
        node2, outcome2 = partial_reduce(f.right_child, diagram_size, assignment)

        if outcome1 and outcome2:
            new_bdd = bdd_manager.apply('and', node1.value, node2.value)
            return BDD(new_bdd), bdd_manager.count(new_bdd) > diagram_size

        return And(node1, node2), False

    raise TypeError(f"{type(f)} not recognized.")