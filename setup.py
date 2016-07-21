from setuptools import setup, find_packages
import numpy as np
import os
package_data = {'radd':['docs/*.md', 'docs/*.txt', 'examples/*.csv', 'examples/*.mp4', 'styles/*.css']}
setup(
    name='RADD',
    version='0.1.5',
    author='Kyle Dunovan, Timothy Verstynen',
    author_email='dunovank@gmail.com',
    url='http://github.com/CoAxLab/radd',
    packages=['radd', 'radd.rl', 'radd.tools', 'radd.examples'],
    package_data=package_data,
    description='RADD (Race Against Drift-Diffusion model) is a python package for fitting & simulating cognitive models of reinforcement learning and decision-making',
    install_requires=['numpy>=1.8.2', 'scipy>=0.16.1', 'matplotlib>=1.4.3', 'seaborn>=0.5.1', 'pandas>=0.15.1', 'lmfit>=0.9.1', 'sklearn>=0.17.1', 'progressbar2>=3.9.3', 'future'],
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
                'Topic :: Scientific/Engineering',
                ],
)
