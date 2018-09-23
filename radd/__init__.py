#!usr/bin/env python
import os
import glob
modules = glob.glob(os.path.dirname(__file__)+"/*.py")
__all__ = [ os.path.basename(f)[:-3] for f in modules]

_package_dir = os.path.dirname(os.path.realpath(__file__))
__version__ = '0.4.0'

def load_example_data():
    import pandas as pd
    _examples_dir = os.path.join(_package_dir, 'docs/examples')
    data_fpath = os.path.join(_examples_dir, 'reactive_example_idx.csv')
    return pd.read_csv(data_fpath).copy()

def load_dpm_animation():
    import io, base64
    from IPython.display import HTML
    _examples_dir = os.path.join(_package_dir, 'docs/examples')
    mov_fpath = os.path.join(_examples_dir, 'anim.mp4')
    video = io.open(mov_fpath, 'r+b').read()
    encoded = base64.b64encode(video)
    data='''<video width="50%" alt="test" loop=1 controls> <source src="data:video/mp4; base64,{0}" type="video/mp4" /> </video>'''.format(encoded.decode('ascii'))
    return HTML(data=data)

def style_notebook():
    from IPython.core.display import HTML
    g3link = "(https://www.github.com/dunovank/jupyter-themes)"
    #print("Notebook Theme: Grade3\n{}".format(g3link))
    _styles_dir = os.path.join(_package_dir, 'styles')
    style = os.path.join(_styles_dir, 'custom.css')
    csscontent = open(style, "r").read()
    return HTML(csscontent)
