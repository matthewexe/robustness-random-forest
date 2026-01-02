from robustness.domain.mappers.rf import rf_to_formula_str
from robustness.domain.psf.parser import parse_psf
from robustness.domain.types import _RF_Type, _PSF_Type
from robustness.domain.logging import get_logger

logger = get_logger(__name__)


def from_rf(rf: _RF_Type) -> _PSF_Type:
    logger.info("Converting random forest to PSF")
    formula = rf_to_formula_str(rf)
    psf = from_formula_str(formula)
    logger.debug("Random forest converted to PSF successfully")
    return psf


def from_formula_str(formula: str) -> _PSF_Type:
    logger.info(f"Parsing formula string of length {len(formula)}")
    psf = parse_psf(formula)
    logger.debug("Formula parsed to PSF successfully")
    return psf
