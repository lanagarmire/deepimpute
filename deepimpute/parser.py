import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="scRNA-seq data imputation using DeepImpute."
    )
    parser.add_argument("inputFile", type=str, help="Path to input data.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./imputed.csv",
        help="Path to output data counts. Default: ./imputed.csv",
    )
    parser.add_argument(
        "--cores", type=int, default=-1, help="Number of cores. Default: all available cores"
    )
    parser.add_argument(
        "--cell-axis",
        type=str,
        choices=["rows", "columns"],
        default="rows",
        help="Cell dimension in the matrix. Default: rows",
    )
    parser.add_argument(
        "--limit",
        type=str,
        default="auto",
        help="Genes to impute (e.g. first 2000 genes). Default: auto",
    )
    parser.add_argument(
        "--minVMR",
        type=float,
        default="0.5",
        help="Min Variance over mean ratio for gene exclusion. Gene with a VMR below ${minVMR} are discarded. Used if --limit is set to 'auto'. Default: 0.5",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1,
        help="Cell subset to speed up training. \
                        Either a ratio (0<x<1) or a cell number (int). Default: 1 (all)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
        help="Learning rate. Default: 0.0001"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size. Default: 64"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=300,
        help="Maximum number of epochs. Default: 500"
    )
    parser.add_argument(
        "--hidden-neurons",
        type=int,
        default=300,
        help="Number of neurons in the hidden dense layer. Default: 256"
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.2,
        help="Dropout rate for the hidden dropout layer (0<rate<1). Default: 0.2"
    )
    parser.add_argument(
        "--output-neurons",
        type=int,
        default=512,
        help="Number of output neurons per sub-network. Default: 512"
    )
    parser.add_argument(
        "--n_pred",
        type=int,
        default=None,
        help="Number of predictors to consider. Consider using this parameter if your RAM is limited or if you have a high number of features. Default: All genes with nonzero VMR"
    )

    parser.add_argument(
        "--policy",
        type=str,
        default='restore',
        help="Whether to restore positive values from the raw dataset or keep the max between the imputed values and the raw values. Choices are ['restore', 'max']. Default: restore"
    )

    args = parser.parse_args()

    return args

