import argparse

from anytree import RenderTree

import robustness.schemas.random_forest as rfschema
import robustness.transformers.rf as rftrans
from robustness.adapters.rf_service import RandomForestService
from robustness.domain.bdd.manager import get_bdd_manager, declare_vars
from robustness.domain.config import Config
from robustness.domain.mappers.psf import from_formula_str
from robustness.domain.mappers.rf import rf_to_formula_str
from robustness.domain.psf.operations import partial_reduce, let, tableau_method


def main():
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

    # Config initialization
    config_cls = Config()
    config_cls.diagram_size = args.diagram_size
    config_cls.prefix_var = args.prefix_var
    config_cls.dataset_name = args.dataset_name
    config_cls.rf_path = args.rf_path

    # Verbose
    if args.verbose:
        print("Verbose mode")

    # Reading random forest
    print("Recovering files...")
    print("Reading files...")
    rf_adapter = RandomForestService(
        model_name=config_cls.dataset_name, base_results_path=config_cls.rf_path
    )
    rf_schema = rfschema.from_json(rf_adapter.get_random_forest(), rfschema.RandomForestSchema)
    rf_model = rftrans.rf_schema_to_model(rf_schema)

    # Creating formula
    psf_str = rf_to_formula_str(rf_model)
    psf = from_formula_str(psf_str)

    # Init bdd manager variables
    declare_vars(get_bdd_manager(), psf)

    # Partial reduce
    print("-"*50)
    print(psf)
    print("-"*50)
    new_psf, _ = partial_reduce(psf, diagram_size=config_cls.diagram_size)
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
    tm = tableau_method(new_psf)
    print(RenderTree(tm.root))


if __name__ == "__main__":
    main()
