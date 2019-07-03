# DeepImpute: an accurate and efficient deep learning method for single-cell RNA-seq data imputation

[![Build Status](https://travis-ci.org/lanagarmire/deepimpute.svg?branch=master)](https://travis-ci.org/lanagarmire/deepimpute)

Arisdakessian, Cedric, Olivier Poirion, Breck Yunits, Xun Zhu, and Lana Garmire.  
"DeepImpute: an accurate, fast and scalable deep neural network method to impute single-cell RNA-Seq data." bioRxiv (2018): 353607"  
https://www.biorxiv.org/content/early/2018/06/22/353607

DeepImpute has been implemented in Python2 and Python3. The recommended version is Python3.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Installing

To install DeepImpute, you only need to download the git repository at https://github.com/lanagarmire/deepimpute and install it using pip:

```bash
git clone https://github.com/lanagarmire/deepimpute
cd deepimpute
pip install --user .
```

### Usage

DeepImpute can be used either on the commandline or as a Python package.

Command line:

```
usage: deepImpute.py [-h] [-o O] [--cores CORES] [--cell-axis {rows,columns}]
                     [--limit LIMIT] [--minVMR MINVMR] [--subset SUBSET]
                     [--learning-rate LEARNING_RATE] [--batch-size BATCH_SIZE]
                     [--max-epochs MAX_EPOCHS]
                     [--hidden-neurons HIDDEN_NEURONS]
                     [--dropout-rate DROPOUT_RATE]
                     [--output-neurons OUTPUT_NEURONS]
                     inputFile
					 
scRNA-seq data imputation using DeepImpute.

positional arguments:
  inputFile             Path to input data (raw counts).

optional arguments:
  -h, --help            show this help message and exit
  -o O                  Path to output data counts. Default: ./
  --cores CORES         Number of cores. Default: 5
  --cell-axis {rows,columns}
                        Cell dimension in the matrix. Default: rows
  --limit LIMIT         Genes to impute (e.g. first 2000 genes). Default: auto
  --subset SUBSET       Cell subset to speed up training. Either a ratio
                        (0<x<1) or a cell number (int). Default: 1 (all)
  --learning-rate LEARNING_RATE
                        Learning rate. Default: 0.0005
  --batch-size BATCH_SIZE
                        Batch size. Default: 64
  --max-epochs MAX_EPOCHS
                        Maximum number of epochs. Default: 300
  --hidden-neurons HIDDEN_NEURONS
                        Number of neurons in the hidden dense layer. Default:
                        300
  --dropout-rate DROPOUT_RATE
                        Dropout rate for the hidden dropout layer (0<rate<1).
  --output-neurons OUTPUT_NEURONS
                        Number of output neurons per sub-network. Default: 500
```

Python package:

```python
from deepimpute.deepImpute import deepImpute

data = pd.read_csv('../examples/test.csv', index_col=0) # dimension = (cells x genes)
imputed = deepImpute(data, NN_lim='auto', n_cores=16, cell_subset=1)
```

A more detailed usage of deepImpute's functionality is available in the iPython Notebook notebook_example.ipynb

### Running the tests

Each file has been validated using a unittest script. They are all available in the test folder.
To run all the tests at once, you can also use the makefile by running `make test`.

## License

Copyright (C) 2018 Cedric Arisdakessian - Released under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
