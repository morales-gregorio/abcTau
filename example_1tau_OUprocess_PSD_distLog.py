# add the path to the abcTau package
import sys
sys.path.append('./abcTau')
# import the package
import abcTau
import numpy as np
from scipy import stats

# stetting the number of cores for each numpy computation in multiprocessing
# uncomment if you don't want numy to use more cores than what specified by multiprocessing
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"


# path for loading and saving data
datasave_path = './example_abc_results/'
dataload_path = './example_data/'

# path and filename to save the intermediate results after running each step
inter_save_direc = './example_abc_results/'
inter_filename = './abc_intermediate_results_psd'

# load real data and define filename_save
filename = 'OU_tau20_mean0_var1_rawData'
filename_save = filename
print(filename)
data_load = np.load(dataload_path + filename + '.npy')

# load data time-series as a numpy array (n_trials * time-points)
data_load = np.load(dataload_path + filename + '.npy')

# select summary statistics metric
summary_metric = 'comp_psd'
normalize = True # if normalize the autocorrelation or PSD

# extract statistics from real data
dt = 1 # temporal resolution of data.
binsize = 1 #  bin-size for binning data and computing the autocorrelation.
disp = None # put the disperssion parameter if computed with grid-search
max_lag = None # only used when using autocorrelation for summary statistics
data_summary, data_mean, data_var, T, n_trials =  abcTau.preprocessing.extract_stats(data_load, dt, binsize,\
                                                                                  summary_metric, normalize, max_lag)



# Define the prior distribution
# for a uniform prior: stats.uniform(loc=x_min,scale=x_max-x_min)
t_min = 0.0 # first timescale
t_max = 300.0
prior_dist = [stats.uniform(loc= t_min, scale = t_max - t_min)]


# select generative model and distance function
generativeModel = 'oneTauOU'
distance_func = 'logarithmic_distance'


# set fitting params
epsilon_0 = 1  # initial error threshold
min_samples = 100 # min samples from the posterior
steps = 60 # max number of iterations
min_acc_rate = 0.01 # minimum acceptance rate to stop the iterations
parallel = True # if parallel processing
n_procs = 20 # number of processor for parallel processing (set to 1 if there is no parallel processing)


# creating model object
class my_model(abcTau.Model):

    #This method initializes the model object.
    def __init__(self):
        pass

    # draw samples from the prior.
    def draw_theta(self):
        theta = []
        for p in self.prior:
            theta.append(p.rvs())
        return theta

    # Choose the generative model (from generative_models)
    # Choose autocorrelation computation method (from basic_functions)
    def generate_data(self, theta):
        # generate synthetic data
        if disp is None:
            syn_data, n_bin_data =  eval('abcTau.generative_models.' + generativeModel + \
                                         '(theta, dt, binsize, T, n_trials, data_mean, data_var)')
        else:
            syn_data, n_bin_data =  eval('abcTau.generative_models.' + generativeModel + \
                                         '(theta, dt, binsize, T, n_trials, data_mean, data_var, disp)')

        # compute the summary statistics
        syn_summary = abcTau.summary_stats.comp_summary(syn_data, summary_metric, normalize, dt, binsize, T,\
                                          n_bin_data, max_lag)
        return syn_summary

    # Computes the summary statistics
    def summary_stats(self, data):
        sum_stat = data
        return sum_stat

    # Choose the method for computing distance (from basic_functions)
    def distance_function(self, data, synth_data):
        if np.nansum(synth_data) <= 0: # in case of all nans return large d to reject the sample
            d = 10**4
        else:
            d = eval('abcTau.distance_functions.' +distance_func + '(data, synth_data)')
        return d


# fit with aABC algorithm for any generative model
abc_results, final_step = abcTau.fit.fit_withABC(my_model, data_summary, prior_dist, inter_save_direc, inter_filename,\
                                                 datasave_path,filename_save, epsilon_0, min_samples, \
                                                 steps, min_acc_rate, parallel, n_procs, disp)