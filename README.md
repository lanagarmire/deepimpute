# DeepImpute: an accurate and efficient deep learning method for single-cell RNA-seq data imputation

DeepImpute has been implemented in Python2 and Python3. The recommended version is Python3.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Installing

To install DeepImpute, you only need to download the git repository at https://github.com/lanagarmire/deepimpute and install it using pip:

```bash
git clone https://github.com/lanagarmire/deepimpute
cd deepimpute
pip install -e .
```

### Usage

DeepImpute can be used either on the commandline or as a Python package.

Command line:

```
usage: deepImpute.py [-h] [-o outputFile] [--ncores NCORES]
                     [--cell-axis {rows,columns}] [--limit LIM]
                     [--subset SUBSET]
                     inputFile

scRNA-seq data imputation using DeepImpute.

positional arguments:
  inputFile             Path to input data.

optional arguments:
  -h, --help            show this help message and exit
  -o outputFile         Path to output data counts. Default: ./
  --cores NCORES       Number of cores. Default: 5
  --cell-axis {rows,columns}
                        Cell dimension in the matrix. Default: rows
  --limit LIM             Genes to impute (e.g. first 2000 genes). Default: auto
  --subset SUBSET       Cell subset to speed up training. Either a ratio
                        (0<x<1) or a cell number (int). Default: 1 (all)
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
