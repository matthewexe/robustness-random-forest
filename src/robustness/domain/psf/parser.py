import robustness.domain.psf as psf
from robustness.domain.logging import get_logger

logger = get_logger(__name__)

"""
Partially Satisfiable Formula Grammar

PSF := True | False | VAR | NOT PSF | PSF AND PSF
NOT := not | Not | ! | ~
AND := and | And | && | &
VAR := \\w+
"""

from lark import Token

grammar = r"""

start: dnf

?dnf: term
    | term OR dnf                   

?term: literal
    | literal AND term

?literal: NOT VARIABLE
        | CONSTANT                  
        | VARIABLE
        | CLASS              

AND.2: "and" | "And" | "AND" | "&&" | "&"
OR.2:  "or"  | "Or"  | "OR"  | "||" | "|"
NOT.2: "not" | "Not" | "NOT" | "!" | "~"


CONSTANT.2: "true" | "false"
CLASS: /c\d+/
VARIABLE: /t_\d+/

%import common.WS
%ignore WS
%ignore "("
%ignore ")"
"""


def ast_to_formula(lark_tree, builder: psf.Builder) -> int:
    if isinstance(lark_tree, Token):
        if lark_tree.type == "CONSTANT":
            return builder.Constant(lark_tree.value == 'true')
        if lark_tree.type == "VARIABLE":
            return builder.Variable(lark_tree.value)
        if lark_tree.type == "CLASS":
            return builder.Class(lark_tree.value)
    else:
        if lark_tree.data == "start":
            return ast_to_formula(lark_tree.children[0], builder)
        if lark_tree.data == "literal":
            child = lark_tree.children[0]
            if child.type == "NOT":
                child = ast_to_formula(lark_tree.children[1], builder)
                return builder.Not(child)
            return ast_to_formula(child, builder)
        if lark_tree.data in {"term", "dnf"}:
            if len(lark_tree.children) == 0:
                return ast_to_formula(lark_tree.children[0], builder)
            left = ast_to_formula(lark_tree.children[0], builder)
            right = ast_to_formula(lark_tree.children[2], builder)
            if lark_tree.data == "term":
                return builder.And(left, right)
            else:
                return builder.Or(left, right)

    raise ValueError(f"{lark_tree.data} unrecognized")


def parse_psf(formula_str: str) -> psf.PSF:
    import lark as l

    logger.info(f"Parsing PSF formula string (length: {len(formula_str)})")
    logger.debug(f"Formula: {formula_str[:200]}..." if len(formula_str) > 200 else f"Formula: {formula_str}")

    parser = l.Lark(grammar=grammar, parser="lalr")

    lark_tree = parser.parse(formula_str)
    builder = psf.Builder()
    root_id = ast_to_formula(lark_tree.children[0], builder)
    logger.info(f"PSF formula parsed successfully. root_id=[{root_id}]")
    return builder.build()
