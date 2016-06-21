#!usr/bin/env python
import os
import glob
import pandas as pd
modules = glob.glob(os.path.dirname(__file__)+"/*.py")
__all__ = [ os.path.basename(f)[:-3] for f in modules]
__version__ = '0.0.11'

examples_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')
def load_example_data():
    data_fpath = os.path.join(examples_dir, 'reactive_example_idx.csv')
    return pd.read_csv(data_fpath).copy()

def css_styling(style='oceans16'):
    from IPython.display import HTML
    style_fpath = os.path.join(examples_dir, style+'.css')
    styles = open(style, "r").read()
    return HTML(styles)
