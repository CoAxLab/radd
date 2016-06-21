#!usr/bin/env python
import os
import glob
import pandas as pd
modules = glob.glob(os.path.dirname(__file__)+"/*.py")
__all__ = [ os.path.basename(f)[:-3] for f in modules]
__version__ = '0.0.10'


def load_example_data():
    examples_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
    data_fpath = os.path.join(examples_dir, 'reactive_example_idx.csv')
    return pd.read_csv(data_fpath).copy()
