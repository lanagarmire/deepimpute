import os
import pandas as pd

PATH = os.path.dirname(os.path.realpath(__file__)) + "/../examples/test."
rawData = pd.read_csv(PATH + "csv", index_col=0)
