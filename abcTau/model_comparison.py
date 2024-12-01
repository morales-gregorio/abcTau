"""
Module for Bayesian model comparison.
"""

import numpy as np

# from generative_models import *
# from basic_functions import *
# from preprocessing import *
# from distance_functions import *
# from summary_stats import *


def model_comp(
    real_data,
    dt,
    binsize,
    max_lag,
    abc_results1,
    final_step1,
    abc_results2,
    final_step2,
    model1,
    model2,
    distance_func,
    summary_metric,
    normalize,
    numSamplesModelComp,
    eval_start=3,
    disp1=None,
    disp2=None,
):
    """Perform Baysian model comparison with ABC fits from model1 and model2.

    Parameters
    -----------
    real_data : nd array
        time-series of continous data, e.g., OU process, (n_trials * n_timepoints)
        or binned spike counts (n_trials * n_bins).
    dt : float
        temporal resolution of data (or binsize of spike counts).
    binsize : float
        bin-size for computing the autocorrelation.
    max_lag : float
        maximum time-lag for computing the autocorrelation.
    abc_results1: object
        output of fitting model1 with aABC algorithm.
    final_step1 : int
        final step of aABC fitting for model1.
    abc_results2: object
        output of fitting model2 with aABC algorithm.
    final_step2 : int
        final step of aABC fitting for model2.
    model1: string
        selected generative model for model1 (from generative models list).
    model2: string
        selected generative model for model2 (from generative models list).
    distance_func: string
        'linear_distance' or 'logarithmic_distance'.
    summary_metric : string
        metric for computing summay statistics ('comp_cc', 'comp_ac_fft', 'comp_psd').
    normalize : string
        if normalize the autocorrelation or PSD.
    numSamplesModelComp: int
        number of samples from posterior distributions to compute the Bayes factor.
    eval_start : int, default 3
        defines the number of smallest errors we ignore before starting CDF computation.
    disp1 : float, default None
        The value of dispersion parameter if computed with the grid search method for model1.
    disp2 : float, default None
        The value of dispersion parameter if computed with the grid search method for model2.

    Returns
    -------
    d1 : 1d array
        distribution of errors (distances) for model1.
    d2 : 1d array
        distribution of errors (distances) for model2.
    cdf1 : 1d array
        CDF of errors for model1.
    cdf2 : 1d array
        CDF of errors for model2.
    err_threshs : 1d array
        error thresholds for which CDFs are computed
    bf : 1d array
        Bayes factors for each error threshold in "err_threshs" (CDF_M2/CDF_M1).
    """
    # extract abc fits
    theta_accepted1 = abc_results1[final_step1 - 1]["theta accepted"]
    theta_accepted2 = abc_results2[final_step2 - 1]["theta accepted"]

    # extract real data statistics
    data_summary, data_mean, data_var, T, n_trials = extract_stats(
        real_data, dt, binsize, summary_metric, normalize, max_lag
    )

    # compute distances
    n_samples_post1 = len(theta_accepted1[0])
    n_samples_post2 = len(theta_accepted2[0])
    print("Computing distances for model1:")
    d1 = gen_model_dist(
        data_summary,
        theta_accepted1,
        numSamplesModelComp,
        n_samples_post1,
        model1,
        distance_func,
        summary_metric,
        normalize,
        dt,
        binsize,
        T,
        n_trials,
        data_mean,
        data_var,
        max_lag,
        disp1,
    )
    print("Computing distances for model2:")
    d2 = gen_model_dist(
        data_summary,
        theta_accepted2,
        numSamplesModelComp,
        n_samples_post2,
        model2,
        distance_func,
        summary_metric,
        normalize,
        dt,
        binsize,
        T,
        n_trials,
        data_mean,
        data_var,
        max_lag,
        disp2,
    )

    # compute CDFs and Bayes factors
    cdf1, cdf2, eval_points, bf = comp_cdf(d1, d2, numSamplesModelComp, eval_start)
    err_threshs = eval_points
    return d1, d2, cdf1, cdf2, err_threshs, bf


def gen_model_dist(
    data_summary,
    theta_accepted,
    numSamplesModelComp,
    numSamplesPosterior,
    model,
    distance_func,
    summary_metric,
    normalize,
    dt,
    binsize,
    T,
    n_trials,
    data_mean,
    data_var,
    max_lag,
    disp=None,
):
    """Compute distances for a given data autocorrelation and generative model.

    Parameters
    -----------
    data_summary : 1d array
        summary statistics of real data (autocorrelation or PSD).
    theta_accepted : nd array
        accepted samples in aABC Posteriors.
    numSamplesModelComp: int
        number of samples from posterior distributions to compute the Bayes factor.
    numSamplesPosterior: int
        number of samples for each parameter in aABC Posteriors.
    model: string
        selected generative model (from generative models list).
    distance_func: string
        'linear_distance' or 'logarithmic_distance'.
    dt : float
        temporal resolution of data (or binsize of spike counts).
    binsize : float
        bin-size for computing the autocorrelation.
    T : float
        duration of trials.
    n_trials : float
        number of trials.
    data_mean : float
        mean value of data.
    data_var : float
        variance of data.
    max_lag : float
        maximum time-lag for computing the autocorrelation.
    disp : float, default None
        The value of dispersion parameter if computed with the grid search method.


    Returns
    -------
    d_all : 1d array
        distribution of errors (distances) for the given generative model.
    """
    d_all = []

    if disp is None:
        for s in range(numSamplesModelComp):
            # select a random sample from the multivariate Posterior
            print("Sample ", s)
            j = np.random.randint(numSamplesPosterior)
            theta = theta_accepted[:, j]

            # generate synthetic data
            syn_data, n_bin_data = eval(
                model + "(theta, dt, binsize, T, n_trials, data_mean, data_var)"
            )
            syn_summary = comp_summary(
                syn_data, summary_metric, normalize, dt, binsize, T, n_bin_data, max_lag
            )

            d = eval(distance_func + "(data_summary, syn_summary)")
            d_all.append(d)

    else:
        for s in range(numSamplesModelComp):
            # select a random sample from the multivariate Posterior
            print("Sample ", s)
            j = np.random.randint(numSamplesPosterior)
            theta = theta_accepted[:, j]

            # generate synthetic data
            syn_data, n_bin_data = eval(
                model + "(theta, dt, binsize, T, n_trials, data_mean, data_var, disp)"
            )
            syn_summary = comp_summary(
                syn_data, summary_metric, normalize, dt, binsize, T, n_bin_data, max_lag
            )

            d = eval(distance_func + "(data_summary, syn_summary)")
            d_all.append(d)

    return d_all


def comp_cdf(d1, d2, num_samples, eval_start=3):
    """Compute CDF of errors for fitted models.

    Parameters
    -----------
    d1 : 1d array
        distribution of errors (distances) for model1.
    d2 : 1d array
        distribution of errors (distances) for model2.
    num_samples: int
        number of samples for each parameter in aABC Posteriors.
    eval_start : int, default 3
        defines the number of smallest errors we ignore before starting CDF computation.


    Returns
    -------
    cdf1 : 1d array
        CDF of errors for model1.
    cdf2 : 1d array
        CDF of errors for model2.
    err_threshs : 1d array
        error thresholds for which CDFs are computed
    bf : 1d array
        Bayes factors for each error threshold in "err_threshs" (CDF_M2/CDF_M1).
    """

    d1_sorted = np.sort(d1)
    d2_sorted = np.sort(d2)
    eval_points = np.sort(
        np.unique(np.concatenate((d1_sorted[eval_start:], d2_sorted[eval_start:])))
    )
    cdf1 = []
    cdf2 = []
    for i in range(len(eval_points)):
        ind1 = np.where(d1_sorted <= eval_points[i])
        if np.size(ind1):
            cdf1.append((np.max(ind1) + 1) / num_samples)
        else:
            cdf1.append(0)
        ind2 = np.where(d2_sorted <= eval_points[i])
        if np.size(ind2):
            cdf2.append((np.max(ind2) + 1) / num_samples)
        else:
            cdf2.append(0)
    bf = np.array(cdf2) / np.array(cdf1)
    return cdf1, cdf2, eval_points, bf
