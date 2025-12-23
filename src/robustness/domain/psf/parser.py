from robustness.domain.psf.model import And, Or, Not, Constant, Formula, Variable

"""
Partially Satisfiable Formula Grammar

PSF := True | False | VAR | NOT PSF | PSF AND PSF
NOT := not | Not | ! | ~
AND := and | And | && | &
VAR := \\w+
"""

from lark import Token


grammar = """

start: dnf

?dnf: term
    | term OR dnf                   

?term: literal
     | literal AND term  

?literal: NOT VARIABLE
        | CONSTANT                  
        | VARIABLE              

AND.2: "and" | "And" | "AND" | "&&" | "&"
OR.2:  "or"  | "Or"  | "OR"  | "||" | "|"
NOT.2: "not" | "Not" | "NOT" | "!" | "~"

CONSTANT.2: "true" | "false"
VARIABLE: /[a-zA-Z_][a-zA-Z0-9_]*/

%import common.WS
%ignore WS
%ignore "("
%ignore ")"
"""


def ast_to_formula(lark_tree) -> Formula:
    if isinstance(lark_tree, Token):
        if lark_tree.type == "CONSTANT":
            return Constant(lark_tree.value == "true")
        if lark_tree.type == "VARIABLE":
            return Variable(lark_tree.value)
    else:
        if lark_tree.data == "start":
            return ast_to_formula(lark_tree.children[0])
        if lark_tree.data == "literal":
            child = lark_tree.children[0]
            if child.type == "NOT":
                child = ast_to_formula(lark_tree.children[1])
                return Not(child)
            return ast_to_formula(child)
        if lark_tree.data in {"term", "dnf"}:
            if len(lark_tree.children) == 0:
                return ast_to_formula(lark_tree.children[0])
            left = ast_to_formula(lark_tree.children[0])
            right = ast_to_formula(lark_tree.children[2])
            if lark_tree.data == "term":
                return And(left, right)
            else:
                return Or(left, right)

    raise TypeError(f"Unkown token {lark_tree}")


def parse_psf(formula_str: str) -> Formula:
    import lark as l

    parser = l.Lark(grammar=grammar, parser="lalr")

    lark_tree = parser.parse(formula_str)  # type: ignore
    return ast_to_formula(lark_tree.children[0])
