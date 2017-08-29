# Bounded Global & Local Optimization

## Stochastic Global Optimization (Basinhopping w/ bounds)
* **set_basinparams()** method gives control over low-level parameters used for global opt
    * **xtol = ftol = tol** (default=1e-20)
    * set basinhopping initial step-size to .1 (adaptive)
        * see HopStep class in radd.fit and get_stepsize_scalars() in radd.theta for parameter-specific step-sizes
    * Sample **nsamples** (default=5000) parameters sets,
    * perform global optimization on best **ninits** (default=5)
        * For each param set, basinhopping performs stochastic search of bounded parameter space
    * continues until **nsuccess** (default=40) steps pass without finding new global minimum
    * parameters with the lowest global minimum (i.e. best of 5)  are then passed to local optimization

## Local Optimization (Nelder-Mead Simplex w/ bounds)
* **set_fitparams()** method gives control over low-level parameters used for local opt
    * local optimization polishes parameter estimates passed from global optimization step
    * **xtol = ftol = tol** (default=1e-30)

# Steps in Fitting Routine

## Step 1. Flat Fits
- All models are initially fit by optimizing the full set of parameters to the "flattened" data (flat meaning the average data collapsing across all conditions of interest). For instance, at this stage fitting the dependent process model involves finding the parameter values for each included parameter that minimizes the cost-function cost function:

  $Cost = \sum [\omega * (\hat{Y} - Y)]^2$


- **$Y$** is an array of observed data (e.g., accuracy, RT quantiles, etc.)


- **$\hat{Y}$** is an equal length array of corresponding model-predicted values given parameters $\theta$


- The error between the predicted and the observed data (**$\hat{Y} - Y$**) is weighted by an equal length array of scalars **$\omega$** proportional to the inverse of the variance in each value of **$Y$**.


- The array of weighted differences is summed, yielding a single cost value equal to the summed squared-error (**$SSE$**).


| Parameters ($\theta$) | Description | Go/Stop |
|:--:|:--|:---:|
| a | Boundary Height | -- |
| tr | Onset-Delay | Go |
| v | Drift-Rate | Go |
| xb | Dynamic Gain | Go |
| ssv | Drift-Rate | Stop |
| sso | Onset-Delay | Stop |

### Step 1a:
- Global optimzation on flat data (average values collapsing across experimental conditions)

### Step 1b:
- Local optimzation using parameters passed from global optimizer as starting values


## Step 2. Conditional Fits
- Conditional models can be fit in which all parameters from **Step 1** are held constant except for one or more designated **conditional** parameters which is free to vary across levels an experimental condition of interest. Global and local optimization are performed by minimizing the cost-function:

    $Cost = \sum_{i=0}^{N_c} [\omega_i * (\hat{Y_i} - Y_i)]^2$


- where $\sum[\omega_i*(\hat{Y_i} - Y_i)]^2$ gives the **$Cost$** for level **$i$** of condition **$C$**


- the total **$Cost$** is equal to the **$SSE$** across all **$N_c$** levels of that condition


- Specifying parameter dependencies is done using **depends_on** --> **{parameter_id : condition_name}**.


- For instance, in Dunovan et al., (2015) subjects performed two versions of a stop-signal task
    - **Baseline ("bsl")**: subjects received equal penalties for not responding on Go trials and executing a response on "Stop-Signal" trials
    - **Penalty ("pnl")**: responding on "Stop-Signal" trials elicits double the penalty for ommitted responses on "Go" trials
    - Different hypotheses about the mechanism underlying behavioral adaptation to assymetric risk of committing different types of errors

- To test the hypothesis that a change in decision threshold **(a)** provides a better account of the data than a change in the drift-rate **(v)**, where **'Cond'** is the name of a column header in your data file


  | How-To |Code |
  |:---:|:---:|
  | initiate boundary model |**model_a = build.Model(data, depends_on={'a':'Cond'})** |
  | initiate drift model  | **model_v = build.Model(data, depends_on={'v':'Cond'})** |
  | fit boundary model | **model_a.optimize()** |
  | fit drift model | **model_v.optimize()** |
  | Is boundary model better? | **model_a.finfo['AIC'] < model_v.finfo['AIC']** |

### Step 2a
- Global optimzation of conditional parameters

### Step 2b:
- Local optimzation of conditional parameters passed from global optimizer

# Initializing and Fitting Models
``` python
# set progress=True to track the model error at each basinhopping step (red)
# as well as the global minimum (green) across all param sets (ninits = 5; see above)
model_a = build.Model(kind='xdpm', data=coxon_data, depends_on={'a':'Cond'}, fit_on='average')
model_a.optimize(plot_fits=True, progress=True)

model_v = build.Model(kind='xdpm', data=coxon_data, depends_on={'v':'Cond'}, fit_on='average')
model_v.optimize(plot_fits=True, progress=True)
```

# Model Comparison
``` python
models = ['Boundary', 'Drift-Rate']
gof_names = ['AIC', 'BIC']

# extract GOF stats from finfo attribute
a_gof = model_a.finfo[gof_names]
v_gof = model_v.finfo[gof_names]

# print GOF stats for both models
print('Boundary GOF:\nAIC = {}\nBIC = {}\n'.format(*a_gof))
print('Drift-Rate GOF:\nAIC = {}\nBIC = {}\n'.format(*v_gof))

# Which model provides a better fit to the data?
aicwinner = models[np.argmin([a_gof[0], v_gof[0]])]
bicwinner = models[np.argmin([a_gof[1], v_gof[1]])]
print('AIC likes {} model'.format(aicwinner))
print('BIC likes {} model'.format(bicwinner))
```

## How to access...

|model information | method used to calculate | how to access|
|--|--|--|
| flat data | **model**.observedDF.mean() | **model**.observed_flat |
| flat weights | **model**.wtsDF.mean() | **model**.flat_wts |
| conditional data | **model**.observedDF.groupby(**condition**).mean()| **model**.observed |
| conditional weights | **model**.wtsDF.groupby(**condition**).mean() |  **model**.cond_wts |


# Troubleshooting Ugly Fits

## Fit to individual subjects
* model = build.Model(data=data, ..., **fit_on**=**'subjects'**)

## Other "kinds" of models...
* Currently only Dependent Process Model **(kind='dpm')** and Independent Race Model **(kind='irace')**
* Tell model to include a Dynamic Bias Signal **('xb')** by adding an **'x'** to the front of model **kind**
* To implement the Dependent Process Model **('irace')**
    * with dynamic bias: model = build.Model(data=data, **kind**=**xdpm'** ... )
    * without: model = build.Model(data=data, **kind**=**dpm'**, ... )
* To implement the Independent Race Model **('irace')**
    * with dynamic bias: model = build.Model(data=data, **kind**=**xirace'** ... )
    * without: model = build.Model(data=data, **kind**=**irace'**, ... )

## Other dependencies...
* Maybe subjects change their boundary height or go onset time across conditions
* Model with dynamic gain free across conditions:
    * model = build.Model(data=data, ... , **depends_on**={**'xb'**: **'Cond'**})
* Model with onset-delay free across conditions:
    * model = build.Model(data=data, ... , **depends_on**={**'tr'**: **'Cond'**})

## Optimization parameters...
 * model.set_basinparams(nsuccess=50, tol=1e-30, ninits=10, nsamples=10000)
 * model.set_fitparams(maxfev=5000, tol=1e-35)
 * Check out the wts vectors for extreme vals
     * Try re-running the fits with an unweighted model (all wts = 1)
         * m = build.Model(data=data, ... weighted=False)
 * Error RTs can be particularly troublesome, sometimes un-shootably so...


# Examine fits
```python
# the fit summary (goodness of fit measures, etc.)
model_v.fitdf

# model predictions
# to save as csv file: model_v.yhatdf.to_csv("save_path", index=False)
# to extract values as numpy ndarray: model_v.yhatdf.loc[:, 'acc':].values
model_v.yhatdf

# best-fit parameter estimates also stored in popt dictionary
model_v.popt
```
