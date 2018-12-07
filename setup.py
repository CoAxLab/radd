from setuptools import setup, find_packages
import numpy as np
import os

package_data = {'radd':['docs/*.md', 'docs/*.txt', 'styles/*.css', 'datasets/eLife15/*.csv', 'datasets/jNeuro18/*.csv', 'docs/examples/*.py', 'docs/examples/*.png', 'docs/examples/*.mp4']}

setup(
    name='RADD',
    version='0.5.0',
    author='Kyle Dunovan, Timothy Verstynen, Jeremy Huang',
    author_email='dunovank@gmail.com',
    url='http://github.com/CoAxLab/radd',
    packages=['radd', 'radd.compiled', 'radd.adapt', 'radd.tools', 'radd.docs', 'radd.datasets', 'radd.docs.examples'],
    package_data=package_data,
    description='RADD (Race Against Drift-Diffusion model) is a python package for fitting & simulating cognitive models of reinforcement learning and decision-making',
    install_requires=['numpy>=1.8.2', 'scipy>=0.16.1', 'matplotlib>=1.4.3', 'seaborn>=0.5.1', 'pandas>=0.15.1', 'lmfit>=0.9.1', 'scikit-learn>=0.17.1', 'progressbar2>=3.9.3', 'numba>=0.30.1', 'pyDOE', 'future'],
    include_dirs = [np.get_include()],
    classifiers=[
                'Environment :: Console',
                'Operating System :: OS Independent',
                'License :: OSI Approved :: BSD License',
                'Intended Audience :: Science/Research',
                'Development Status :: 3 - Alpha',
                'Programming Language :: Python',
                'Programming Language :: Python :: 2',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.2',
                'Programming Language :: Python :: 3.4',
                'Programming Language :: Python :: 3.6',
                'Topic :: Scientific/Engineering',
                ]
)
