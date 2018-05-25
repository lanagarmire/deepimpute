import pandas as pd
from deepimpute.multinet import MultiNet


def deepImpute(
    data,
    NN_lim="auto",
    n_cores=10,
    cell_subset=None,
    imputed_only=False,
    seed=0,
    **NN_params
):

    multi = MultiNet(n_cores=n_cores, seed=seed, **NN_params)
    multi.fit(data, NN_lim=NN_lim, cell_subset=cell_subset)
    return multi.predict(data, imputed_only=imputed_only)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="scRNA-seq data imputation using DeepImpute."
    )
    parser.add_argument("inputFile", type=str, help="Path to input data.")
    parser.add_argument(
        "-o",
        metavar="outputFile",
        type=str,
        help="Path to output data counts. Default: ./",
    )
    parser.add_argument(
        "--cores", type=int, default=5, help="Number of cores. Default: 5"
    )
    parser.add_argument(
        "--cell-axis",
        type=str,
        choices=["rows", "columns"],
        default=0,
        help="Cell dimension in the matrix. Default: rows",
    )
    parser.add_argument(
        "--limit",
        type=str,
        default="auto",
        help="Genes to impute (e.g. first 2000 genes). Default: auto",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1,
        help="Cell subset to speed up training. \
                        Either a ratio (0<x<1) or a cell number (int). Default: 1 (all)",
    )

    args = parser.parse_args()

    data = pd.read_csv(args.inputFile, index_col=0)
    if args.cell_axis == "columns":
        data = data.T
    imputed = deepImpute(
        data, n_cores=args.cores, NN_lim=args.limit, cell_subset=args.subset
    )
    imputed.to_csv(args.outputFile)
