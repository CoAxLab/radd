from setuptools import setup
import numpy as np

setup(
    name='RADD',
    version='0.0.4',
    author='Kyle Dunovan, Timothy Verstynen',
    author_email='dunovank@gmail.com',
    url='http://github.com/CoAxLab/radd',
    packages=['radd', 'radd.tools', 'radd.rl'],
    description='RADD (Race Against Drift-Diffusion) is a python package for fitting & simulating cognitive models of reinforcement learning and decision-making',
    install_requires=['NumPy >=1.8.2', 'SciPy >= 0.16.1', 'matplotlib >= 1.4.3', 'seaborn>=0.5.1', 'pandas >= 0.12.0', 'lmfit>=0.9.0'],
    setup_requires=['NumPy >=1.8.2', 'SciPy >= 0.16.1', 'matplotlib >= 1.4.3', 'seaborn>=0.5.1', 'pandas >= 0.12.0', 'lmfit>=0.9.0'],
    include_package_data=True,
    include_dirs = [np.get_include()],
)
