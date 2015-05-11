This repository is associated with the manuscript "Competing basal-ganglia pathways determine the difference between stopping and deciding not to go" and contains code and documentation for the dependent process model as well as example data and optimized parameter sets for running simulations.

This demo is not meant to be a comprehensive walkthrough of all analyses and modeling procedures reported in the paper but meant to provide some additional examples of the model and to allow users to experiment with the model by running simulations.

This code should not be used for any type of clinical purposes.


#Files in the "demo/" directory:

###IPython Notebook with various examples
* RADD_Demo.ipynb - http://nbviewer.ipython.org/github/CoAxLab/radd_demo/blob/master/demo/RADD_Demo.ipynb

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
