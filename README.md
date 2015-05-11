This repository is associated with the manuscript "Competing basal-ganglia pathways determine the difference between stopping and deciding not to go" and contains code and documentation for the dependent process model as well as example data and optimized parameter sets for running simulations.

This demo is not meant to be a comprehensive walkthrough of all analyses and modeling procedures reported in the paper but meant to provide some additional examples of the model and to allow users to experiment with the model by running simulations.

This code should not be used for any type of clinical purposes.


#Files in the "demo/" directory:

###IPython Notebook with various examples & simulations
* [RADD_Demo.ipynb](http://nbviewer.ipython.org/github/CoAxLab/radd_demo/blob/master/demo/RADD_Demo.ipynb)

###Example data (9 subjects)
* pro_nogo.csv - probability of nogo decisions in proactive task 9 (subjects) x 6 (Go trial probability)
* pro_rt.csv - mean 'go' RT in proactive task 9 (subjects) x 6 (Go trial probability)
* reB_rt.csv - mean correct 'go' RT in reactive baseline task 9 (subjects)
* reB_stop.csv - mean stop accuracy in reactive baseline task 9 (subjects) x 5 (SSD)
* reC_rt.csv - mean correct 'go' RT in reactive caution task 9 (subjects)
* reC_stop.csv - mean stop accuracy in reactive caution task 9 (subjects) x 5 (SSD)


###Parameter sets for running simulations
* reB_theta.csv - mean optimized parameter set for bootstrapped fits to reactive baseline data
* pro_theta.csv - mean optimized parameter set for bootstrapped fits to proactive data


#Tutorial

###import libraries
```python
#from within cloned radd_demo directory
import *
import numpy as np
import pandas as pd

```
###read data
```python
nogos=pd.read_csv("pro_nogo.csv", index_col=0)
prort=pd.read_csv("pro_rt.csv", index_col=0)
```

###plot data across Go trial probability
```python

#mean 'no-go' decision curve
mean_nogo = nogos.mean().values
#plot 'no-go' decision curve
axp = vis.scurves([mean_nogo], task='Pro', sxdata=[nogos])

#RT(s) -> RT(ms)
rts = prort.mean().values[1:]*1000
rterr = prort.sem().values[1:]*1.96*1000
#plot RTs
axrt = vis.prort(bars=rts, berr=rterr)
```

###simulate proactive data (drift-bias)
```python
#read in parameters file
protheta=pd.read_csv("pro_theta.csv", index_col=0)

#convert to dictionary and extract all drift-rates to list
ptheta, vlist = utils.get_params_from_flatp(protheta)

#simulate effect of changing execution drift-rate across Go trial probability
nogo_sim, prort_sim = fitpro.simple_prosim(ptheta, bias_list=vlist, bias='v')
```

###plot the results
```python
labels=['data', 'drift-bias']
axp = vis.scurves([mean_nogo, nogo_sim], task='Pro', sxdata=[nogos], labels=labels)
axrt = vis.prort(bars=rts, berr=rterr, lines=[rt_sim], labels=labels)
```

###experiment with different parameter values
* v = drift-rate
* z = starting point
* a = boundary height
* t = onset time

```python
# lower boundary height by 10%
ptheta['a'] = ptheta['a']*.9
nogo_sim, prort_sim = fitpro.simple_prosim(ptheta, bias_list=vlist, bias='v')

axp = vis.scurves([mean_nogo, nogo_sim], task='Pro', sxdata=[nogos], labels=labels)
axrt = vis.prort(bars=rts, berr=rterr, lines=[rt_sim], labels=labels)
```
