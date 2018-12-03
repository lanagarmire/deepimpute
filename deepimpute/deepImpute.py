import pandas as pd
from deepimpute.multinet import MultiNet


def deepImpute(
    data,
    NN_lim="auto",
    n_cores=10,
    cell_subset=1,
    imputed_only=False,
    restore_pos_values=True,
    seed=0,
    **NN_params
):

    multi = MultiNet(n_cores=n_cores, seed=seed, **NN_params)
    multi.fit(data, NN_lim=NN_lim, cell_subset=cell_subset)
    return multi.predict(data, imputed_only=imputed_only,restore_pos_values=restore_pos_values)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="scRNA-seq data imputation using DeepImpute."
    )
    parser.add_argument("inputFile", type=str, help="Path to input data.")
    parser.add_argument(
        "-o",
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
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
        help="Learning rate. Default: 0.0005"
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
        help="Maximum number of epochs. Default: 300"
    )
    parser.add_argument(
        "--hidden-neurons",
        type=int,
        default=300,
        help="Number of neurons in the hidden dense layer. Default: 300"
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.5,
        help="Dropout rate for the hidden dropout layer (0<rate<1)."
    )
    parser.add_argument(
        "--nb-corr",
        type=int,
        default=20,
        help="Number of input gene per target gene. Default: 20"
    )
    parser.add_argument(
        "--output-neurons",
        type=int,
        default=500,
        help="Number of output neurons per sub-network. Default: 500"
    )

    args = parser.parse_args()

    data = pd.read_csv(args.inputFile, index_col=0)
    if args.cell_axis == "columns":
        data = data.T

    NN_params = {'dims': [args.nb_corr,args.output_neurons],
                 'learning_rate': args.learning_rate,
                 'batch_size': args.batch_size,
                 'max_epochs': args.max_epochs,
                 'layers': [
                     {"label": "dense", "activation": "relu", "nb_neurons": args.hidden_neurons},
                     {"label": "dropout", "activation": "dropout", "rate": args.dropout_rate},
                     {"label": "dense", "activation": "relu"}] }
    
    imputed = deepImpute(
        data, n_cores=args.cores, NN_lim=args.limit, cell_subset=args.subset, **NN_params
    )
    imputed.to_csv(args.o)
