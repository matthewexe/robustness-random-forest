import argparse

from anytree import RenderTree

import robustness.schemas.random_forest as rfschema
import robustness.transformers.rf as rftrans
from robustness.adapters.rf_service import RandomForestService
from robustness.domain.bdd.manager import get_bdd_manager, declare_vars
from robustness.domain.config import Config
from robustness.domain.logging import get_logger
from robustness.domain.mappers.psf import from_formula_str
from robustness.domain.mappers.rf import rf_to_formula_str
from robustness.domain.psf.operations import partial_reduce, let, tableau_method

logger = get_logger(__name__)


def main():
    logger.info("Starting robustness application")
    parser = argparse.ArgumentParser(description="Esempio di CLI completa con argparse")

    # Output filename
    parser.add_argument(
        "-o",
        "--output",
        help="Nome del file di output (default: out.txt)",
        default="out.txt",
    )

    # Random forest options
    parser.add_argument(
        "--rf-path",
        help="Folder that contains random forests info",
        default="./Random_Forest_Aeon_Univariate/results",
    )
    parser.add_argument(
        "-dn",
        "--dataset-name",
        help="Aeon dataset name used to train random forest",
        default="Meat",
    )
    parser.add_argument("-dd", "--diagram-size", default=50, type=int, help="Max diagram size during partial reduction")
    parser.add_argument("--prefix-var", default="t_", help="Prefix variable in random forest")
    # Boolean flag
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Attiva modalità verbose"
    )


    # Argument parsing
    args = parser.parse_args()
    logger.debug(f"Parsed arguments: {args}")

    # Config initialization
    config_cls = Config()
    config_cls.diagram_size = args.diagram_size
    config_cls.prefix_var = args.prefix_var
    config_cls.dataset_name = args.dataset_name
    config_cls.rf_path = args.rf_path
    logger.info(f"Configuration initialized: dataset={config_cls.dataset_name}, diagram_size={config_cls.diagram_size}")
    logger.debug(f"Full config: prefix_var={config_cls.prefix_var}, rf_path={config_cls.rf_path}")

    # Verbose
    if args.verbose:
        print("Verbose mode")
        logger.debug("Verbose mode enabled")

    # Reading random forest
    print("Recovering files...")
    print("Reading files...")
    logger.info(f"Loading random forest model from {config_cls.rf_path}")
    rf_adapter = RandomForestService(
        model_name=config_cls.dataset_name, base_results_path=config_cls.rf_path
    )
    logger.debug(f"Created RandomForestService adapter for dataset: {config_cls.dataset_name}")
    rf_schema = rfschema.from_json(rf_adapter.get_random_forest(), rfschema.RandomForestSchema)
    logger.debug(f"Loaded random forest schema with {len(rf_schema)} trees")
    rf_model = rftrans.rf_schema_to_model(rf_schema)
    logger.info(f"Converted schema to random forest model with {len(rf_model)} trees")

    # Creating formula
    logger.info("Creating PSF formula from random forest model")
    psf_str = rf_to_formula_str(rf_model)
    logger.debug(f"Generated formula string (length: {len(psf_str)})")
    psf = from_formula_str(psf_str)
    logger.info("Parsed PSF formula successfully")

    # Init bdd manager variables
    logger.info("Initializing BDD manager and declaring variables")
    declare_vars(get_bdd_manager(), psf)
    logger.debug("BDD manager variables declared")

    # Partial reduce
    print("-"*50)
    print(psf)
    print("-"*50)
    logger.info(f"Starting partial reduction with diagram_size={config_cls.diagram_size}")
    new_psf, _ = partial_reduce(psf, diagram_size=config_cls.diagram_size)
    logger.info("Partial reduction completed successfully")
    logger.debug(f"Reduced PSF: {new_psf}")
    print("-"*50)
    print(new_psf)
    print("-"*50)

    # Test let
    # print("-"*50)
    # print(new_psf)
    # print("-"*50)
    # print("-"*50)
    # test_let = let(new_psf, reduce=True, **{"t_175": False, "t_343": True})
    # print(test_let)
    # print("-"*50)

    # Tableau method
    logger.info("Applying tableau method")
    tm = tableau_method(new_psf)
    logger.debug(f"Tableau tree generated with root: {tm.root}")
    print(RenderTree(tm.root))
    logger.info("Robustness application completed successfully")


if __name__ == "__main__":
    main()
