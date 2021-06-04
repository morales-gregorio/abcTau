## abcTau

abcTau is a Python package for unbiased estimation of timescales from autocorrelations or power spectrums using adaptive Approximate Bayesian Computations (aABC). This method overcomes the statistical bias in autocorrelations of finite data samples, which undermines the accuracy of conventional methods based on direct fitting of the autocorrelation with exponential decay  functions.  abcTau overcomes the bias by fitting the sample autocorrelation or power spectrum with a generative model based on a mixture of Ornstein-Uhlenbeck (OU) processes. This way it accounts for finite sample size and noise in data and returns a posterior distribution of timescales, that quantifies the uncertainty of estimates and can be used to compare alternative hypotheses about the dynamics of the underlying process. This method can be applied to any time-series data such as spike-counts in neuroscience.

The details of the method are explained in:  
Zeraati, R., Engel, T. A. & Levina, A. A flexible Bayesian framework for unbiased estimation of timescales. bioRxiv 2020.08.11.245944 (2021). https://www.biorxiv.org/content/10.1101/2020.08.11.245944v2  
Please cite this reference when you use this package for a scientific publication.

You can find Demos on how to use abcTau for estimating timescales and performing Bayesian model comparison in the  Jupyter Notebook Tutorials 1 and 2. These tutorials contain examples already used in figures 1-3 of the preprint above. The example data to test the package are available in "example_data" folder. The example outputs are available in "example_abc_results" and "example_modelComparison". Three example python scripts are also avaiable for running the package on a cluster with parallel processing. You can use the "check_expEstimates" function from the the "preprocessing" module to check the bias in timescales estimated from exponential fits on your data.


For a recent application of this method see:   
Zeraati, R., Shi, Y., Steinmetz, N. A., Gieselmann, M. A., Thiele, A., Moore, T., Levina, A. & Engel, T. A. Attentional modulation of intrinsic timescales in visual cortex and spatial networks. bioRxiv 2021.05.17.444537 (2021). https://www.biorxiv.org/content/10.1101/2021.05.17.444537v1


The basis of aABC algorithm in this package is adopted from a previous implementation (originally developed in Python 2):
Robert C. Morehead and Alex Hagen. A Python package for Approximate Bayesian Computation (2014). https://github.com/rcmorehead/simpleabc. 


## Operating system
- macOs
- Windows
- Linux


## Dependencies
- Python >= 3.7.1
- Numpy >= 1.15.4 
- Scipy >= 1.1.0 


## Installation
For the current version you need to clone this repository and add the package path manually to your Python script or Jupyter Notebook:
```
import sys
sys.path.append('./abcTau')
```
You can see the working examples in the Jupyter Notebook Tutorials 1 and 2. Typical time for installation is less 10 sec.
For the final version of the package we will provide the "pip install" option.


## Development
The object oriented implementation of the package allows users to easily replace any function, including generative models, summary statistic computations, distance functions, etc., with their customized functions to better describe statistics of their data. You can send us your customized generative models to be added directly to the package and create a larger database of generative models available for different applications (contact: research@roxanazeraati.org)


### List of available summary statistics (check "summary_stats.py" for details):
Ordered from fastest to slowest fitting:
- comp_psd: computing the power spectral density
- comp_ac_fft: computing autocorrelation using FFT to speed up (more biased for direct fitting) 
- comp_cc: computing autocorrelation in the time domain (less biased for direct fitting)


### List of available generative models (check "generative_models.py" for details):
- oneTauOU: one-timescale OU process 
- twoTauOU: two-timescale OU process 
- oneTauOU_oscil: one-timescale OU process with an additive oscillation
- oneTauOU_poissonSpikes: one-timescale inhomogenous Poisson process
- oneTauOU_gammaSpikes: one-timescale spiking process with gamma distributed spike-counts and predefined dispersion
- oneTauOU_gaussianSpikes: one-timescale spiking process with Gaussian distributed spike-counts and predefined dispersion
- twoTauOU_poissonSpikes: two-timescale inhomogenous Poisson process
- twoTauOU_gammaSpikes: two-timescale spiking process with gamma distributed spike-counts and predefined dispersion
- twoTauOU_gaussianSpikes: two-timescale spiking process with Gaussian distributed spike-counts and predefined dispersion
- twoTauOU_gammaSpikes_withDispersion: two-timescale spiking process with gamma distributed spike-counts
- twoTauOU_gaussianSpikes_withDispersion: two-timescale spiking process with Gaussian distributed spike-counts


### List of available distance functions (check "distance_functions.py" for details):
- linear_distance
- logarithmic_distance
