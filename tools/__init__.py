#!usr/bin/env python
import os
import glob
modules = glob.glob(os.path.dirname(__file__) + "/*.py")
__all__ = [os.path.basename(f)[:-3] for f in modules]

"""
RUNTIME TESTS:
------------------------------------------------------------------
10 iterations of simulating 10,000 trials:
==================================================================

fitre + RADD: first loop iterates 10 runs, second loop iterates
conditions, ssre_minfunc iterates over 5 SSD's with a for loop
------------------------------------------------------------------
# get param dict and make an array for drift-rates for two cond


      p = {k:v for k,v in redf_store['rebsl_boot_popt'].items()}
      v_cond = np.array([p['v']*1.05,  p['v']*.95])

      ***********************************************************

      '%''%'timeit
      # 10 runs
      for i in range(10):
            # 2 Conditions
            for i in range(2):
                  #update drift-rate, sim 5000 trials
                  p['v'] = v_cond[i]
                  yhat2 = fitre.ssre_minfunc(p, y2, ntrials=5000)


      <OUTPUT> 1 loops, best of 3: 1min 21s per loop

      ***********************************************************


==================================================================
fit: first loop iterates 10 runs, recost calls fit.simulate_full
which vectorizes Go and Stop processes across 2 conditions & 5 SSD
------------------------------------------------------------------


      # include drift-rate for two conditions in param dict
      p = {k:v for k,v in redf_store['rebsl_boot_popt'].items()}
      p['v0'], p['v1'] = np.array([p['v']*1.05,  p['v']*.95])

      ***********************************************************

      '%''%'timeit
      # 10 runs
      for i in range(10):
            # 2 Conditions x 5000 trials
            yhat = fit.recost(p, y, ntrials=10000)


      <OUTPUT> 1 loops, best of 3: 7.82 s per loop

      ***********************************************************


"""
