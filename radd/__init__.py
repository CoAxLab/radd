#!usr/bin/env python
import os
import glob
import pandas as pd
modules = glob.glob(os.path.dirname(__file__)+"/*.py")
__all__ = [ os.path.basename(f)[:-3] for f in modules]
__version__ = '0.1.0'
_examples_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'examples')

def load_example_data():
    data_fpath = os.path.join(_examples_dir, 'reactive_example_idx.csv')
    return pd.read_csv(data_fpath).copy()

def load_dpm_animation():
    import io, base64
    from IPython.display import HTML
    mov_fpath = os.path.join(_examples_dir, 'anim.mp4')
    video = io.open(mov_fpath, 'r+b').read()
    encoded = base64.b64encode(video)
    data='''<video width="40%" alt="test" loop=1 controls> <source src="data:video/mp4; base64,{0}" type="video/mp4" /> </video>'''.format(encoded.decode('ascii'))
    return HTML(data=data)

def style_notebook():
    from IPython.core.display import HTML
    print("""Notebook Theme: Grade3\nmore at github.com/dunovank/jupyter-themes""")
    css_path = os.path.join(_examples_dir, 'custom.css')
    styles = open(css_path, "r").read()
    return HTML(styles)
