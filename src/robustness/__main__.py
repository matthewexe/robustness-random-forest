import argparse

from networkx.drawing.nx_agraph import write_dot

import robustness.transformers.rf as rftrans
from robustness.adapters.rf_service import RandomForestService
from robustness.domain.config import Config
from robustness.domain.logging import get_logger
from robustness.domain.mappers.psf import from_formula_str
from robustness.domain.mappers.rf import rf_to_formula_str
from robustness.domain.psf.operations import partial_reduce, tableau_method, write_psf, robustness, \
    generate_robustness_graph
from robustness.domain.utils.logs import init_log_dirs

logger = get_logger(__name__)


def main():
    try:
        logger.info("Robustness application completed successfully")
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
        parser.add_argument("-dd", "--diagram-size", default=50, type=int,
                            help="Max diagram size during partial reduction")
        parser.add_argument("--prefix-var", default="t_", help="Prefix variable in random forest")
        parser.add_argument("--debug", help="Debug mode", action='store_true')
        parser.add_argument("--log-graphs", help="Log graphs", action='store_true')
        parser.add_argument("--sample-group", help="Group of sample to test robustness on (default: 1)", default=1, type=int)
        parser.add_argument("--sample-id", help="ID of sample to test robustness on (default: 0)", default=0, type=int)
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
        config_cls.debug_mode = args.debug or False
        config_cls.log_graphs = (args.log_graphs or False) or config_cls.debug_mode

        logger.info(
            f"Configuration initialized: {config_cls}")

        if config_cls.log_graphs:
            init_log_dirs()


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

        # Loading random forest
        logger.debug(f"Created RandomForestService adapter for dataset: {config_cls.dataset_name}")
        rf_schema = rf_adapter.random_forest()
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
        logger.debug("BDD manager variables declared")

        # First partial reduction
        logger.info(f"Starting partial reduction with diagram_size={config_cls.diagram_size}")
        if config_cls.log_graphs:
            write_psf(psf, "initial_psf.dot")
        new_psf, _ = partial_reduce(psf, diagram_size=config_cls.diagram_size)
        logger.info("Partial reduction completed successfully")
        logger.debug(f"Reduced PSF: {new_psf}")

        # Tableau method
        logger.info("Applying tableau method")
        tm = tableau_method(new_psf)
        write_dot(tm, "tableau.dot")
        logger.debug(f"Tableau tree {tm} generated with root: {tm.root}")
        logger.info("Robustness application completed successfully")

        # leaf_id = list(tm.leaves)[0]
        # tree = tm.nodes[leaf_id]['tree']
        # bdd_id = tree.root
        # bdd = tree.nodes[bdd_id]['value']
        # get_bdd_manager().dump("bdd.json", roots=[bdd])

        # Test sample
        endpoints_schema = rf_adapter.endpoints()
        endpoints = rftrans.endpoints_to_model(endpoints_schema)
        sample_schema = rf_adapter.sample(args.sample_group, args.sample_id)
        sample = rftrans.sample_schema_to_model(args.sample_group, args.sample_id, sample_schema)
        sample_robustness = robustness(tm, sample, endpoints)
        logger.info(f"Robustness of {sample} is: {sample_robustness}")
        if config_cls.log_graphs:
            generate_robustness_graph(tm, sample, endpoints)

    except Exception as e:
        logger.error(f"An error occurred during robustness application: {e}")
        raise e
    finally:
        from robustness.domain.bdd.manager import cleanup_bdd_manager
        cleanup_bdd_manager()


if __name__ == "__main__":
    main()
