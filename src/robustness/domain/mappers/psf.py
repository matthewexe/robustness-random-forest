from robustness.domain.mappers.rf import rf_to_formula_str
from robustness.domain.psf.parser import parse_psf
from robustness.domain.types import _RF_Type, _PSF_Type


def from_rf(rf: _RF_Type) -> _PSF_Type:
    formula = rf_to_formula_str(rf)
    return from_formula_str(formula)


def from_formula_str(formula: str) -> _PSF_Type:
    return parse_psf(formula)
