import argparse

from robustness.models.psf import PSF, parse_psf


def main():
    # Creazione del parser principale
    parser = argparse.ArgumentParser(description="Esempio di CLI completa con argparse")

    # Argomento opzionale con valore di default
    parser.add_argument(
        "-o",
        "--output",
        help="Nome del file di output (default: out.txt)",
        default="out.txt",
    )

    parser.add_argument("--rf-path", help="Folder that contains random forests info")

    # Flag booleano
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Attiva modalità verbose"
    )

    # Parsing degli argomenti
    args = parser.parse_args()

    # Gestione verbose
    if args.verbose:
        print("Modalità verbose attiva")


if __name__ == "__main__":
    # main()
    print("Starting")
    test_formula = "not a"
    psf = parse_psf(test_formula)
    print(psf)
