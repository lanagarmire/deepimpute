import pandas as pd

from deepimpute.parser import parse_args
from deepimpute.multinet import MultiNet

def deepImpute(**kwargs):

    args = parse_args()

    for key, value in kwargs.items():
        setattr(args, key, value)
    
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

    multi = MultiNet(**NN_params)
    multi.fit(data, NN_lim=args.limit, cell_subset=args.subset, minVMR=args.minVMR, n_pred=args.n_pred)

    imputed = multi.predict(data, imputed_only=False, policy=args.policy)

    if args.output is not None:
        imputed.to_csv(args.output)
    else:
        return imputed

if __name__ == "__main__":
    deepImpute()
