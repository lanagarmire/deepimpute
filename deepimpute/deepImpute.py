def deepImpute(
        data,
        NN_lim="auto",
        cell_subset=1,
        imputed_only=False,
        policy="restore",
        minVMR=0.5,
        **NN_params
):
    from deepimpute.multinet import MultiNet

    multi = MultiNet(**NN_params)
    multi.fit(data, NN_lim=NN_lim, cell_subset=cell_subset, minVMR=minVMR)
    return multi.predict(data, imputed_only=imputed_only, policy=policy)

if __name__ == "__main__":
    import argparse
    import pandas as pd

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
        "--cores", type=int, default=-1, help="Number of cores. Default: all available cores"
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

    args = parser.parse_args()

    data = pd.read_csv(args.inputFile, index_col=0)
    if args.cell_axis == "columns":
        data = data.T

    NN_params = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'ncores': args.cores,
        'sub_outputdim': args.output_neurons,
        'architecture': [
            {"type": "dense", "activation": "relu", "neurons": args.hidden_neurons},
            {"type": "dropout", "activation": "dropout", "rate": args.dropout_rate}]
    }
    
    imputed = deepImpute(
        data,
        NN_lim=args.limit,
        cell_subset=args.subset,
        minVMR=args.minVMR,
        **NN_params
    )

    if args.cell_axis == "columns":
        imputed = imputed.T
        
    imputed.to_csv(args.o)
