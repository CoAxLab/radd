#!usr/bin/env python
import os
import glob
import pandas as pd
modules = glob.glob(os.path.dirname(__file__)+"/*.py")
__all__ = [ os.path.basename(f)[:-3] for f in modules]
__version__ = '0.0.18'

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
    data='''<video width="50%" alt="test" loop=1 controls> <source src="data:video/mp4; base64,{0}" type="video/mp4" /> </video>'''.format(encoded.decode('ascii'))
    return HTML(data=data)
