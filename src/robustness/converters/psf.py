from robustness.converters.rf import rf_to_formula_str
from robustness.models.types import _RF_Type, _PSF_Type
from robustness.models.psf import PSF
from robustness.parser.psf import parse_psf


def from_rf(rf: _RF_Type) -> _PSF_Type:
    formula = rf_to_formula_str(rf)
    return from_formula_str(formula)


def from_formula_str(formula: str) -> _PSF_Type:
    tree = parse_psf(formula)
    return PSF(formula_str=formula, formula=tree)
