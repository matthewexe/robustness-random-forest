import robustness.domain.psf as psf
from robustness.domain.config import Config
from robustness.domain.logging import get_logger
from robustness.domain.utils.formula import is_class

logger = get_logger(__name__)
config = Config()

"""
Partially Satisfiable Formula Grammar

PSF := True | False | VAR | NOT PSF | PSF AND PSF
NOT := not | Not | ! | ~
AND := and | And | && | &
VAR := \\w+
"""

grammar = r"""
    ?start: iff

    ?iff: xor
        | iff "~" xor     -> iff_op

    ?xor: or
        | xor "^" or      -> xor_op

    ?or: and
       | or "|" and       -> or_op

    ?and: not
        | and "&" not     -> and_op

    ?not: "!" not         -> not_op
        | atom

    ?atom: variable     -> var
         | true         -> true_const
         | false        -> false_const
         | "(" iff ")"  -> parens

    variable: /[a-zA-Z_][a-zA-Z0-9_]*/
    true: "\\T"
    false: "\\F"

    %import common.WS
    %ignore WS
"""


def ast_to_formula(lark_tree, builder: psf.Builder) -> int:
    match lark_tree.data:
        case "iff_op":
            left = ast_to_formula(lark_tree.children[0], builder)
            right = ast_to_formula(lark_tree.children[1], builder)
            return builder.Iff(left, right)
        case "xor_op":
            left = ast_to_formula(lark_tree.children[0], builder)
            right = ast_to_formula(lark_tree.children[1], builder)
            return builder.Xor(left, right)
        case "or_op":
            left = ast_to_formula(lark_tree.children[0], builder)
            right = ast_to_formula(lark_tree.children[1], builder)
            return builder.Or(left, right)
        case "and_op":
            left = ast_to_formula(lark_tree.children[0], builder)
            right = ast_to_formula(lark_tree.children[1], builder)
            return builder.And(left, right)
        case "not_op":
            child = ast_to_formula(lark_tree.children[0], builder)
            return builder.Not(child)
        case "parens":
            return ast_to_formula(lark_tree.children[0], builder)
        case "var":
            var_name = lark_tree.children[0].children[0].value
            return builder.Variable(var_name) if not is_class(var_name) else builder.Class(var_name)
        case "true_const":
            return builder.Constant(True)
        case "false_const":
            return builder.Constant(False)
        case _:
            logger.warning(f"Unrecognized Lark tree node: {lark_tree.data}")
            return ast_to_formula(lark_tree.children[0], builder)



def parse_psf(formula_str: str) -> psf.PSF:
    import lark as l

    logger.info(f"Parsing PSF formula string (length: {len(formula_str)})")
    logger.debug(f"Formula: {formula_str[:200]}..." if len(formula_str) > 200 else f"Formula: {formula_str}")

    parser = l.Lark(grammar=grammar)

    lark_tree = parser.parse(formula_str)
    if config.log_graphs:
        # AST to DOT
        from lark.tree import pydot__tree_to_dot
        pydot__tree_to_dot(lark_tree, "logs/psf/initial_psf_ast.dot")

    builder = psf.Builder()
    root_id = ast_to_formula(lark_tree.children[0], builder)
    logger.info(f"PSF formula parsed successfully. root_id=[{root_id}]")
    return builder.build()
