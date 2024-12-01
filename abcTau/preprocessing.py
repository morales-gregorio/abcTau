"""
Module for extracting statistics from data and running the preprocessing function.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as stats
from basic_functions import binData
from summary_stats import comp_cc, comp_summary

from generative_models import *

# from distance_functions import *


def extract_stats(data, dt, binsize, summary_metric, normalize, max_lag=None):
    """Extract required statistics from data for ABC fitting.

    Parameters
    -----------
    data : nd array
        time-series of continous data, e.g., OU process, (n_trials * numTimePoints)
        or binned spike counts (n_trials * n_bins).
    dt : float
        temporal resolution of data (or binsize of spike counts).
    binsize : float
        bin-size for computing the autocorrelation. It should be a multiple of dt.
    summStat : string
        metric for computing summay statistics ('comp_cc', 'comp_ac_fft', 'comp_psd').
    normalize : string
        if normalize the autocorrelation or PSD.
    max_lag : float, default None
        maximum time-lag when using 'comp_cc' summary statistics.


    Returns
    -------
    summary_nonNorm : 1d array
        non normalized autocorrelation or PSD depending on chosen summary statistics.
    data_mean : float
        mean of data (computed in the unit of binsize)
    data_mean : float
        variance of data (computed in the unit of binsize)
    T : float
        duration of each trial in data
    n_trials : float
        number of trials in data
    binsize : float
        bin-size used for computing the autocorrlation.
    max_lag : float
        maximum time-lag used for computing the autocorrelation.

    """
    # extract duration and number of trials
    n_trials, numTimePoints = np.shape(data)
    T = numTimePoints * dt

    # bin data
    binsData = np.arange(0, T + binsize, binsize)
    n_bin_data = len(binsData) - 1
    binned_data = binData(data, [n_trials, n_bin_data]) * dt

    # compute mean and variance
    data_mean = np.mean(binned_data)
    data_var = comp_cc(binned_data, binned_data, 1, binsize, n_bin_data)[0]

    #     older version, had problems when dt>1
    #     bin_var = 1
    #     binsData_var =  np.arange(0, T + bin_var, bin_var)
    #     numBinData_var = len(binsData_var)-1
    #     binned_data_var = binData(data, [n_trials,numBinData_var]) * dt
    #     data_mean = np.mean(binned_data_var)/bin_var
    #     data_var = comp_cc(binned_data_var, binned_data_var, 1, bin_var, numBinData_var)[0]

    summary = comp_summary(
        binned_data, summary_metric, normalize, dt, binsize, T, n_bin_data, max_lag
    )

    return summary, data_mean, data_var, T, n_trials


def compute_spikesDisperssion_twoTau(
    ac_data,
    theta,
    dt,
    binsize,
    T,
    n_trials,
    data_mean,
    data_var,
    min_disp,
    max_disp,
    jump_disp,
    borderline_jump,
    n_iterations,
):
    """
    Compute the disperssion parameter of spike counts from the autocorrelation of
    a doubly stochastic process with two timescale using grid search method.

    Parameters
    -----------
    ac_data : 1d array
        autocorrelation of data.
    theta : 1d array
        [timescale1, timescale2, coefficient for timescale1].
    dt : float
        temporal resolution for the OU process generation.
    binsize : float
        bin-size for binning data and computing the autocorrelation.
    T : float
        duration of trials.
    n_trials : float
        number of trials.
    data_mean : float
        mean value of the OU process (average of firing rate).
    data_var : float
        variance of the OU process (variance of firing rate).
    min_disp : float
        minimum value of disperssion for grid search.
    max_disp : float
        maximum value of disperssion for grid search.
    jump_disp : float
        resolution of grid search.
    borderline_jump : int
        is used when the intial range of dispersion grid search is initially small,
        defines number of "jump_disps" to be searched below or above the initial range.

    Returns
    -------
    disp_estim : float
        estimated value of disperssion.

    """
    disp_range = np.arange(min_disp, max_disp, jump_disp)
    max_lag = 2

    min_border = 0
    max_border = 0
    border_line = 1

    while border_line == 1:
        error_all = []
        if min_border or max_border:
            disp_range = np.arange(min_disp, max_disp, jump_disp)

        for disp in disp_range:
            print(disp)
            error_sum = 0
            for _ in range(n_iterations):
                data_syn, n_bin_data = twoTauOU_gammaSpikes(
                    theta, dt, binsize, T, n_trials, data_mean, data_var, disp
                )
                ac_syn = comp_cc(data_syn, data_syn, max_lag, binsize, n_bin_data)
                error_sum = error_sum + abs(ac_syn[1] - ac_data[1])

            error_all.append(error_sum)
        error_all = np.array(error_all)
        disp_estim = disp_range[np.argmin((error_all))]

        if disp_estim == disp_range[0]:
            min_border = 1
            min_disp, max_disp = (
                min_disp - borderline_jump * jump_disp,
                min_disp + jump_disp,
            )

        elif disp_estim == disp_range[-1]:
            max_border = 1
            min_disp, max_disp = (
                max_disp - jump_disp,
                max_disp + borderline_jump * jump_disp,
            )

        else:
            border_line = 0

    return disp_estim


def compute_spikesDisperssion_oneTau(
    ac_data,
    theta,
    dt,
    binsize,
    T,
    n_trials,
    data_mean,
    data_var,
    min_disp,
    max_disp,
    jump_disp,
    borderline_jump,
    n_iterations=200,
):
    """Compute the disperssion parameter of spike counts from the autocorrelation of
    a doubly stochastic process with one timescale using grid search method.

    Parameters
    -----------
    ac_data : 1d array
        autocorrelation of data.
    theta : 1d array
        [timescale].
    dt : float
        temporal resolution for the OU process generation.
    binsize : float
        bin-size for binning data and computing the autocorrelation.
    T : float
        duration of trials.
    n_trials : float
        number of trials.
    data_mean : float
        mean value of the OU process (average of firing rate).
    data_var : float
        variance of the OU process (variance of firing rate).
    min_disp : float
        minimum value of disperssion for grid search.
    max_disp : float
        maximum value of disperssion for grid search.
    jump_disp : float
        resolution of grid search.
    borderline_jump : int
        is used when the intial range of dispersion grid search is initially small,
        defines number of "jump_disps" to be searched below or above the initial range.

    Returns
    -------
    disp_estim : float
        estimated value of disperssion.

    """
    disp_range = np.arange(min_disp, max_disp, jump_disp)
    max_lag = 2

    min_border = 0
    max_border = 0
    border_line = 1

    while border_line == 1:
        error_all = []
        if min_border or max_border:
            disp_range = np.arange(min_disp, max_disp, jump_disp)

        for disp in disp_range:
            print(disp)
            error_sum = 0
            for _ in range(n_iterations):
                data_syn, n_bin_data = oneTauOU_gammaSpikes(
                    theta, dt, binsize, T, n_trials, data_mean, data_var, disp
                )
                ac_syn = comp_cc(data_syn, data_syn, max_lag, binsize, n_bin_data)
                error_sum = error_sum + abs(ac_syn[1] - ac_data[1])

            error_all.append(error_sum)
        error_all = np.array(error_all)
        disp_estim = disp_range[np.argmin((error_all))]

        if disp_estim == disp_range[0]:
            min_border = 1
            min_disp, max_disp = (
                min_disp - borderline_jump * jump_disp,
                min_disp + jump_disp,
            )

        elif disp_estim == disp_range[-1]:
            max_border = 1
            min_disp, max_disp = (
                max_disp - jump_disp,
                max_disp + borderline_jump * jump_disp,
            )

        else:
            border_line = 0

    return disp_estim


def check_estimates(
    theta,
    dt,
    binsize,
    T,
    n_trials,
    data_mean,
    data_var,
    max_lag,
    n_timescales,
    n_iterations=500,
    plot_it=1,
):
    """
    Preprocessing function to check if timescales from exponential fits are reliable for the given data.

    Parameters
    -----------
    theta : 1d array
        timescales fitted by exponential functions on real data autocorrelations
        [timescale1, timescale2, coefficient for timescale1] or [timescale]
    dt : float
        temporal resolution for the OU process generation.
    binsize : float
        bin-size for binning data and computing the autocorrelation.
    T : float
        duration of trials.
    n_trials : float
        number of trials.
    data_mean : float
        mean value of the OU process (average of firing rate).
    data_var : float
        variance of the OU process (variance of firing rate).
    max_lag : float
        maximum time-lag for computing and fitting autocorrelations
    n_timescales: 1 or 2
        number of timescales to fit
    n_iterations: float, default 500
        number iterations to generate synthetic data and fit with exponentials
    plot_it: boolean
        if plot the distributions.

    Returns
    -------
    taus_bs : nd array
        array of estimated timescales from parametric bootstrapping.
    taus_bs_corr: nd array
        array of bootstrap-corrected timescales from exponential fits.
    err: nd array
        Bootstrap-error for each timescale (in percentage)

    """
    if n_timescales > 2:
        raise ValueError("Function is not designed for more than two timescales.")

    if n_timescales == 2:
        tau1_exp = []
        tau2_exp = []
        for _ in range(n_iterations):
            data_syn, _ = twoTauOU(theta, dt, binsize, T, n_trials, data_mean, data_var)
            ac_syn = comp_ac_fft(data_syn)
            lm = round(max_lag / binsize)
            ac_syn = ac_syn[0:lm]

            # fit exponentials
            xdata = np.arange(0, max_lag, binsize)
            ydata = ac_syn
            popt, _ = curve_fit(double_exp, xdata, ydata, maxfev=2000)
            taus = np.sort(popt[1:3])
            tau1_exp.append(taus[0])
            tau2_exp.append(taus[1])
        taus_bs = np.array([tau1_exp, tau2_exp])

        # compute the bootstrap error and do bias correction
        tau1_bs_corr = tau1_exp + 2 * (theta[0] - np.mean(tau1_exp))
        x1 = np.mean(tau1_exp)
        x2 = theta[0]
        err1 = int(((x2 - x1) / x1) * 100)
        tau2_bs_corr = tau2_exp + 2 * (theta[1] - np.mean(tau2_exp))
        x1 = np.mean(tau2_exp)
        x2 = theta[1]
        err2 = int(((x2 - x1) / x1) * 100)

        taus_bs_corr = np.array([tau1_bs_corr, tau2_bs_corr])
        err = np.array([err1, err2])

        print("first timescale: ", str(err1) + "% bootstrap-error")
        print("second timescale: ", str(err2) + "% bootstrap-error")
        print("The true errors from the ground truths can be larger")

        if plot_it:
            plt.figure(figsize=(20, 6))
            ax = plt.subplot(121)
            plt.hist(tau1_exp, color="m", label="Parametric bootstrap", density=True)
            plt.hist(
                tau1_bs_corr,
                ec="m",
                fc="w",
                label="Bootstrap bias-corrected",
                density=True,
            )
            plt.axvline(theta[0], color="c", label="Direct fit")

            plt.xlabel("Timescale")
            plt.ylabel("Probability density")

            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")

            ax = plt.subplot(122)
            plt.hist(tau2_exp, color="m", label="Parametric bootstrap", density=True)
            plt.hist(
                tau2_bs_corr,
                ec="m",
                fc="w",
                label="Bootstrap bias-corrected",
                density=True,
            )
            plt.axvline(theta[1], color="c", label="Direct fit")

            plt.xlabel("Timescale")
            plt.legend(
                frameon=False,
                loc="upper right",
                bbox_to_anchor=(1.7, 0.95),
                handlelength=0.7,
                handletextpad=0.3,
            )

            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")

    if n_timescales == 1:
        # XXX Massive code repetition here
        tau_exp = []
        for _ in range(n_iterations):
            data_syn, _ = oneTauOU(theta, dt, binsize, T, n_trials, data_mean, data_var)
            ac_syn = comp_ac_fft(data_syn)
            lm = round(max_lag / binsize)
            ac_syn = ac_syn[0:lm]

            # fit exponentials
            xdata = np.arange(0, max_lag, binsize)
            ydata = ac_syn
            popt, _ = curve_fit(single_exp, xdata, ydata, maxfev=2000)
            tau_exp.append(popt[1])
        taus_bs = np.array(tau_exp)
        taus_bs_corr = taus_bs + 2 * (theta[0] - np.mean(taus_bs))
        x1 = np.mean(taus_bs)
        x2 = theta[0]
        err = int(((x2 - x1) / x1) * 100)
        print(str(err) + "% bootstrap-error")
        print("The true error from the ground truth can be larger")

        if plot_it:
            plt.figure(figsize=(9, 5))
            ax = plt.subplot(111)
            plt.hist(taus_bs, color="m", label="Parametric bootstrap", density=True)
            plt.hist(
                taus_bs_corr,
                ec="m",
                fc="w",
                label="Bootstrap bias-corrected",
                density=True,
            )
            plt.axvline(theta[0], color="c", label="Direct fit")

            plt.xlabel("Timescale")
            plt.ylabel("Probability density")
            plt.legend(
                frameon=False,
                loc="upper right",
                bbox_to_anchor=(1.6, 0.95),
                handlelength=0.7,
                handletextpad=0.3,
            )

            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.yaxis.set_ticks_position("left")
            ax.xaxis.set_ticks_position("bottom")

    return taus_bs, taus_bs_corr, err


def fit_twoTauExponential(ac, binsize, max_lag):
    """Fit the autocorrelation with a double exponential using non-linear least squares method.

    Parameters
    -----------
    ac : 1d array
        autocorrelation.
    binsize : float
        bin-size used for computing the autocorrelation.
    max_lag : float
        maximum time-lag  used for computing the autocorrelation.

    Returns
    -------
    popt : 1d array
        optimal values for the parameters:
        [amplitude, timescale1, timescale2, weight of the timescale1].
    pcov: 2d array
        estimated covariance of popt. The diagonals provide the variance of the parameter estimate.

    """
    # fit exponentials
    xdata = np.arange(0, max_lag, binsize)
    ydata = ac
    popt, pcov = curve_fit(double_exp, xdata, ydata, maxfev=2000)
    return popt, pcov


def fit_oneTauExponential(ac, binsize, max_lag):
    """
    Fit the autocorrelation with a single exponential using non-linear least squares method.

    Parameters
    -----------
    ac : 1d array
        autocorrelation.
    binsize : float
        bin-size used for computing the autocorrelation.
    max_lag : float
        maximum time-lag  used for computing the autocorrelation.

    Returns
    -------
    popt : 1d array
        optimal values for the parameters: [amplitude, timescale].
    pcov: 2d array
        estimated covariance of popt. The diagonals provide the variance of the parameter estimate.

    """
    # fit exponentials
    xdata = np.arange(0, max_lag, binsize)
    ydata = ac
    popt, pcov = curve_fit(single_exp, xdata, ydata, maxfev=2000)
    return popt, pcov
