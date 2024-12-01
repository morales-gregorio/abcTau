"""
Module containing different generative models
"""

import numpy as np
from scipy import stats
from basic_functions import OU_gen, binData, gamma_sp, gaussian_sp

# XXX This entire file is a catastrophe, each function repeats all the code from the previous ones!
# XXX Only minor differences between all these functions, could be kwargs!


def oneTauOU(theta, dt, binsize, T, n_trials, data_mean, data_var):
    """Generate an OU process with a single timescale.

    Parameters
    -----------
    theta : 1d array
        [timescale].
    dt : float
        temporal resolution for OU process generation.
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

    Returns
    -------
    syn_data : nd array
        array of generated OU process (n_trials * int(T/binsize)).
    n_bin_data : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau = np.array(theta[0])

    # setting params for OU
    v = 1
    D = v / tau
    binned_data = np.arange(0, T + binsize, binsize)
    n_bin_data = len(binned_data) - 1

    # generate OU
    ou_all = OU_gen(tau, D, dt, T, n_trials)
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check > 10**10:  # check for nan values
        return np.zeros((n_trials, n_bin_data)), n_bin_data

    # fit mean and var
    ou_std = np.sqrt(data_var)
    ou_mean = data_mean
    ou_all = ou_std * ou_all + ou_mean

    # bin rate
    syn_data = binData(ou_all, [n_trials, n_bin_data]) * dt
    return syn_data, n_bin_data


def twoTauOU(theta, dt, binsize, T, n_trials, data_mean, data_var):
    """Generate a two-timescales OU process.

    Parameters
    -----------
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

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (n_trials * int(T/binsize)).
    n_bin_data : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau1 = np.array(theta[0])
    tau2 = np.array(theta[1])
    coeff = np.array(theta[2])

    # setting the params of OU
    v = 1
    D1 = v / tau1
    D2 = v / tau2
    binned_data = np.arange(0, T + binsize, binsize)
    n_bin_data = len(binned_data) - 1

    # generate OU
    ou_all1 = OU_gen(tau1, D1, dt, T, n_trials)
    ou_all2 = OU_gen(tau2, D2, dt, T, n_trials)
    ou_all = np.sqrt(coeff) * ou_all1 + np.sqrt(1 - coeff) * ou_all2
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check > 10**10:  # check for all-nan values
        return np.zeros((n_trials, n_bin_data)), n_bin_data

    # fit mean and var
    ou_std = np.sqrt(data_var)
    ou_mean = data_mean
    ou_all = ou_std * ou_all + ou_mean

    # bin rate
    syn_data = binData(ou_all, [n_trials, n_bin_data]) * dt
    return syn_data, n_bin_data


def oneTauOU_oscil(theta, dt, binsize, T, n_trials, data_mean, data_var):
    """Generate a one-timescale OU process with an additive oscillation.

    Parameters
    -----------
    theta : 1d array
        [timescale of OU, frequency of oscillation, coefficient for OU].
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

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (n_trials * int(T/binsize)).
    n_bin_data : int
        number of bins/samples per trial (required for computing autocorrelation).

    """

    # load params
    tau = np.array(theta[0])
    f = np.array(theta[1])
    coeff = np.array(theta[2])

    # setting params for OU
    v = 1
    D = v / tau
    binned_data = np.arange(0, T + binsize, binsize)
    binsData_sin = np.arange(0, T, dt)
    n_bin_data = len(binned_data) - 1

    # generate OU + oscil
    ou_all = OU_gen(tau, D, dt, T, n_trials)
    time_mat = np.tile(binsData_sin, (n_trials, 1))
    phases = np.random.rand(n_trials, 1) * 2 * np.pi
    oscil = np.sqrt(2) * np.sin(phases + 2 * np.pi * 0.001 * f * time_mat)
    data = np.sqrt(1 - coeff) * oscil + np.sqrt(coeff) * ou_all

    # fit mean and var
    ou_std = np.sqrt(data_var)
    ou_mean = data_mean
    data_meanVar = ou_std * data + ou_mean

    # bin rate
    syn_data = binData(data_meanVar, [n_trials, n_bin_data]) * dt
    return syn_data, n_bin_data


def oneTauOU_twooscil(theta, dt, binsize, T, n_trials, data_mean, data_var):
    """Generate a one-timescale OU process with two additive oscillation.

    Parameters
    -----------
    theta : 1d array
        [timescale of OU, frequency of oscillation1, frequency of oscillation2, coefficient for oscillation1, coefficient for oscillation2].
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

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (n_trials * int(T/binsize)).
    n_bin_data : int
        number of bins/samples per trial (required for computing autocorrelation).

    """

    # load params
    tau = np.array(theta[0])
    f1 = np.array(theta[1])
    f2 = np.array(theta[2])
    coeff1 = np.array(theta[3])
    coeff2 = np.array(theta[4])

    # setting params for OU
    v = 1
    D = v / tau
    binned_data = np.arange(0, T + binsize, binsize)
    binsData_sin = np.arange(0, T, dt)
    n_bin_data = len(binned_data) - 1

    # generate OU + oscil
    ou_all = OU_gen(tau, D, dt, T, n_trials)
    time_mat = np.tile(binsData_sin, (n_trials, 1))
    phases = np.random.rand(n_trials, 1) * 2 * np.pi
    oscil1 = np.sqrt(2) * np.sin(phases + 2 * np.pi * 0.001 * f1 * time_mat)
    phases = np.random.rand(n_trials, 1) * 2 * np.pi
    oscil2 = np.sqrt(2) * np.sin(phases + 2 * np.pi * 0.001 * f2 * time_mat)
    data = (
        np.sqrt(coeff1) * oscil1 + np.sqrt(coeff2) * oscil2 + np.sqrt(1 - coeff1 - coeff2) * ou_all
    )

    # fit mean and var
    ou_std = np.sqrt(data_var)
    ou_mean = data_mean
    data_meanVar = ou_std * data + ou_mean

    # bin rate
    syn_data = binData(data_meanVar, [n_trials, n_bin_data]) * dt
    return syn_data, n_bin_data


def oneTauOU_poissonSpikes(theta, dt, binsize, T, n_trials, data_mean, data_var):
    """Generate a one-timescale process with spike counts sampled from a Gaussian distribution.
    Assuming that disperssion parameter (fano factor) of spike generation function is known.

    Parameters
    -----------
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
        mean of the spike counts.
    data_var : float
        variance of the spike counts.


    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (n_trials * int(T/binsize)).
    n_bin_data : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau = np.array(theta[0])

    # setting the params of OU
    v = 1
    D = v / tau

    ou_std = np.sqrt(data_var - data_mean) / dt  # law of total variance
    ou_mean = data_mean / dt  # law of total expectation
    binned_data = np.arange(0, T + binsize, binsize)
    n_bin_data = len(binned_data) - 1

    # generate OU
    ou_all = OU_gen(tau, D, dt, T, n_trials)

    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check > 10**10:  # check for all-nan values
        return np.zeros((n_trials, n_bin_data)), n_bin_data

    # fit mean and var
    ou_all = ou_std * ou_all + ou_mean
    ou_all[ou_all < 0] = 0

    # bin rate and generate spikes
    rate_sum = binData(ou_all, [n_trials, n_bin_data]) * dt
    syn_data = np.random.poisson(rate_sum)
    return syn_data, n_bin_data


def oneTauOU_gammaSpikes(theta, dt, binsize, T, n_trials, data_mean, data_var, disp):
    """Generate a one-timescale process with spike counts sampled from a Gamma distribution.
    Assuming that disperssion parameter (fano factor) of spike generation function is known.

    Parameters
    -----------
    theta : 1d array
        [timescale].
    dt : float
        temporal resolution for the OU process generation. dt <= binsize.
    binsize : float
        bin-size for binning data and computing the autocorrelation
        (should be the same as observed data and a multiple of dt).
    T : float
        duration of trials.
    n_trials : float
        number of trials.
    data_mean : float
        mean of the spike counts, e.g., is computed from spike counts in the unit of binsize.
    data_var : float
        variance of the spike counts, e.g., is computed from spike counts in the unit of binsize.
    disp : float
        disperssion parameter (fano factor) of spike generation function.

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (n_trials * int(T/binsize)).
    n_bin_data : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau = np.array(theta[0])

    # setting the params of OU
    v = 1
    D = v / tau

    ou_std = np.sqrt(data_var - disp * data_mean) / binsize  # law of total variance
    ou_mean = data_mean / binsize  # law of total expectation
    binned_data = np.arange(0, T + binsize, binsize)
    n_bin_data = len(binned_data) - 1

    # generate OU
    ou_all = OU_gen(tau, D, dt, T, n_trials)

    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check > 10**10:  # check for all-nan values
        return np.zeros((n_trials, n_bin_data)), n_bin_data

    # fit mean and var
    ou_all = ou_std * ou_all + ou_mean
    ou_all[ou_all < 0] = 0

    # bin rate and generate spikes
    rate_sum = binData(ou_all, [n_trials, n_bin_data]) * dt
    syn_data = gamma_sp(rate_sum, disp)
    return syn_data, n_bin_data


def oneTauOU_gaussianSpikes(theta, dt, binsize, T, n_trials, data_mean, data_var, disp):
    """Generate a one-timescale process with spike counts sampled from a Gaussian distribution.
    Assuming that disperssion parameter (fano factor) of spike generation function is known.

    Parameters
    -----------
    theta : 1d array
        [timescale].
    dt : float
        temporal resolution for the OU process generation. dt <= binsize.
    binsize : float
        bin-size for binning data and computing the autocorrelation
        (should be the same as observed data and a multiple of dt).
    T : float
        duration of trials.
    n_trials : float
        number of trials.
    data_mean : float
        mean of the spike counts, e.g., is computed from spike counts in the unit of binsize.
    data_var : float
        variance of the spike counts, e.g., is computed from spike counts in the unit of binsize.
    disp : float
        disperssion parameter (fano factor) of spike generation function.

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (n_trials * int(T/binsize)).
    n_bin_data : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau = np.array(theta[0])

    # setting the params of OU
    v = 1
    D = v / tau

    ou_std = np.sqrt(data_var - disp * data_mean) / binsize  # law of total variance
    ou_mean = data_mean / binsize  # law of total expectation
    binned_data = np.arange(0, T + binsize, binsize)
    n_bin_data = len(binned_data) - 1

    # generate OU
    ou_all = OU_gen(tau, D, dt, T, n_trials)

    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check > 10**10:  # check for all-nan values
        return np.zeros((n_trials, n_bin_data)), n_bin_data

    # fit mean and var
    ou_all = ou_std * ou_all + ou_mean
    ou_all[ou_all < 0] = 0

    # bin rate and generate spikes
    rate_sum = binData(ou_all, [n_trials, n_bin_data]) * dt
    syn_data = gaussian_sp(rate_sum, disp)
    return syn_data, n_bin_data


def oneTauOU_gammaSpikes_withDispersion(theta, dt, binsize, T, n_trials, data_mean, data_var):
    """Generate a one-timescale process with spike counts sampled from a Gamma distribution.
    disperssion parameter (fano factor) of spike generation function is fitted with ABC.

    Parameters
    -----------
    theta : 1d array
        [timescale1, disperssion_parameter].
    dt : float
        temporal resolution for the OU process generation. dt <= binsize.
    binsize : float
        bin-size for binning data and computing the autocorrelation
        (should be the same as observed data and a multiple of dt).
    T : float
        duration of trials.
    n_trials : float
        number of trials.
    data_mean : float
        mean of the spike counts, e.g., is computed from spike counts in the unit of binsize.
    data_var : float
        variance of the spike counts, e.g., is computed from spike counts in the unit of binsize.

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (n_trials * int(T/binsize)).
    n_bin_data : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau1 = np.array(theta[0])
    disp = np.array(theta[1])

    # setting the params of OU
    v = 1
    D1 = v / tau1
    ou_std = np.sqrt(data_var - disp * data_mean) / binsize  # law of total variance
    ou_mean = data_mean / binsize  # law of total expectation
    binned_data = np.arange(0, T + binsize, binsize)
    n_bin_data = len(binned_data) - 1

    # generate OU
    ou_all = OU_gen(tau1, D1, dt, T, n_trials)
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check > 10**10:  # check for all-nan values
        return np.zeros((n_trials, n_bin_data)), n_bin_data

    # fit mean and var
    ou_all = ou_std * ou_all + ou_mean
    ou_all[ou_all < 0] = 0

    # bin rate and generate spikes
    rate_sum = binData(ou_all, [n_trials, n_bin_data]) * dt
    syn_data = gamma_sp(rate_sum, disp)
    return syn_data, n_bin_data


def oneTauOU_gaussianSpikes_withDispersion(theta, dt, binsize, T, n_trials, data_mean, data_var):
    """Generate a one-timescale process with spike counts sampled from a Gamma distribution.
    disperssion parameter (fano factor) of spike generation function is fitted with ABC.

    Parameters
    -----------
    theta : 1d array
        [timescale1, disperssion_parameter].
    dt : float
        temporal resolution for the OU process generation. dt <= binsize.
    binsize : float
        bin-size for binning data and computing the autocorrelation
        (should be the same as observed data and a multiple of dt).
    T : float
        duration of trials.
    n_trials : float
        number of trials.
    data_mean : float
        mean of the spike counts, e.g., is computed from spike counts in the unit of binsize.
    data_var : float
        variance of the spike counts, e.g., is computed from spike counts in the unit of binsize.

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (n_trials * int(T/binsize)).
    n_bin_data : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau1 = np.array(theta[0])
    disp = np.array(theta[1])

    # setting the params of OU
    v = 1
    D1 = v / tau1
    ou_std = np.sqrt(data_var - disp * data_mean) / binsize  # law of total variance
    ou_mean = data_mean // binsize  # law of total expectation
    binned_data = np.arange(0, T + binsize, binsize)
    n_bin_data = len(binned_data) - 1

    # generate OU
    ou_all = OU_gen(tau1, D1, dt, T, n_trials)
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check > 10**10:  # check for all-nan values
        return np.zeros((n_trials, n_bin_data)), n_bin_data

    # fit mean and var
    ou_all = ou_std * ou_all + ou_mean
    ou_all[ou_all < 0] = 0

    # bin rate and generate spikes
    rate_sum = binData(ou_all, [n_trials, n_bin_data]) * dt
    syn_data = gaussian_sp(rate_sum, disp)
    return syn_data, n_bin_data


def twoTauOU_poissonSpikes(theta, dt, binsize, T, n_trials, data_mean, data_var):
    """Generate a two-timescales process with spike counts sampled from a Poisson distribution.

    Parameters
    -----------
    theta : 1d array
        [timescale1, timescale2, coefficient for timescale1].
    dt : float
        temporal resolution for the OU process generation. dt <= binsize.
    binsize : float
        bin-size for binning data and computing the autocorrelation
        (should be the same as observed data and a multiple of dt).
    T : float
        duration of trials.
    n_trials : float
        number of trials.
    data_mean : float
        mean of the spike counts, e.g., is computed from spike counts in the unit of binsize.
    data_var : float
        variance of the spike counts, e.g., is computed from spike counts in the unit of binsize.

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (n_trials * int(T/binsize)).
    n_bin_data : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau1 = np.array(theta[0])
    tau2 = np.array(theta[1])
    coeff = np.array(theta[2])

    # setting the params of OU
    v = 1
    D1 = v / tau1
    D2 = v / tau2
    ou_std = np.sqrt(data_var - data_mean) / binsize  # law of total variance
    ou_mean = data_mean / binsize  # law of total expectation
    binned_data = np.arange(0, T + binsize, binsize)
    n_bin_data = len(binned_data) - 1

    # generate OU
    ou_all1 = OU_gen(tau1, D1, dt, T, n_trials)
    ou_all2 = OU_gen(tau2, D2, dt, T, n_trials)
    ou_all = np.sqrt(coeff) * ou_all1 + np.sqrt(1 - coeff) * ou_all2
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check > 10**10:  # check for all-nan values
        return np.zeros((n_trials, n_bin_data)), n_bin_data

    # fit mean and var
    ou_all = ou_std * ou_all + ou_mean
    ou_all[ou_all < 0] = 0

    # bin rate and generate spikes
    rate_sum = binData(ou_all, [n_trials, n_bin_data]) * dt
    syn_data = np.random.poisson(rate_sum)
    return syn_data, n_bin_data


def twoTauOU_gammaSpikes(theta, dt, binsize, T, n_trials, data_mean, data_var, disp):
    """Generate a two-timescales process with spike counts sampled from a Gamma distribution.
    Assuming that disperssion parameter (fano factor) of spike generation function is known.

    Parameters
    -----------
    theta : 1d array
        [timescale1, timescale2, coefficient for timescale1].
    dt : float
        temporal resolution for the OU process generation. dt <= binsize.
    binsize : float
        bin-size for binning data and computing the autocorrelation
        (should be the same as observed data and a multiple of dt).
    T : float
        duration of trials.
    n_trials : float
        number of trials.
    data_mean : float
        mean of the spike counts, e.g., is computed from spike counts in the unit of binsize.
    data_var : float
        variance of the spike counts, e.g., is computed from spike counts in the unit of binsize.
    disp : float
        disperssion parameter (fano factor) of spike generation function.

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (n_trials * int(T/binsize)).
    n_bin_data : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau1 = np.array(theta[0])
    tau2 = np.array(theta[1])
    coeff = np.array(theta[2])

    # setting the params of OU
    v = 1
    D1 = v / tau1
    D2 = v / tau2
    ou_std = np.sqrt(data_var - disp * data_mean) / binsize  # law of total variance
    ou_mean = data_mean / binsize  # law of total expectation
    binned_data = np.arange(0, T + binsize, binsize)
    n_bin_data = len(binned_data) - 1

    # generate OU
    ou_all1 = OU_gen(tau1, D1, dt, T, n_trials)
    ou_all2 = OU_gen(tau2, D2, dt, T, n_trials)
    ou_all = np.sqrt(coeff) * ou_all1 + np.sqrt(1 - coeff) * ou_all2
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check > 10**10:  # check for all-nan values
        return np.zeros((n_trials, n_bin_data)), n_bin_data

    # fit mean and var
    ou_all = ou_std * ou_all + ou_mean
    ou_all[ou_all < 0] = 0

    # bin rate and generate spikes
    rate_sum = binData(ou_all, [n_trials, n_bin_data]) * dt
    syn_data = gamma_sp(rate_sum, disp)
    return syn_data, n_bin_data


def twoTauOU_gaussianSpikes(theta, dt, binsize, T, n_trials, data_mean, data_var, disp):
    """Generate a two-timescales process with spike counts sampled from a Guassion distribution.
    Assuming that disperssion parameter (fano factor) of spike generation function is known.

    Parameters
    -----------
    theta : 1d array
        [timescale1, timescale2, coefficient for timescale1].
    dt : float
        temporal resolution for the OU process generation. dt <= binsize.
    binsize : float
        bin-size for binning data and computing the autocorrelation
        (should be the same as observed data and a multiple of dt).
    T : float
        duration of trials.
    n_trials : float
        number of trials.
    data_mean : float
        mean of the spike counts, e.g., is computed from spike counts in the unit of binsize.
    data_var : float
        variance of the spike counts, e.g., is computed from spike counts in the unit of binsize.
    disp : float
        disperssion parameter (fano factor) of spike generation function.

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (n_trials * int(T/binsize)).
    n_bin_data : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau1 = np.array(theta[0])
    tau2 = np.array(theta[1])
    coeff = np.array(theta[2])

    # setting the params of OU
    v = 1
    D1 = v / tau1
    D2 = v / tau2
    ou_std = np.sqrt(data_var - disp * data_mean) / binsize  # law of total variance
    ou_mean = data_mean / binsize  # law of total expectation
    binned_data = np.arange(0, T + binsize, binsize)
    n_bin_data = len(binned_data) - 1

    # generate OU
    ou_all1 = OU_gen(tau1, D1, dt, T, n_trials)
    ou_all2 = OU_gen(tau2, D2, dt, T, n_trials)
    ou_all = np.sqrt(coeff) * ou_all1 + np.sqrt(1 - coeff) * ou_all2
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check > 10**10:  # check for all-nan values
        return np.zeros((n_trials, n_bin_data)), n_bin_data

    # fit mean and var
    ou_all = ou_std * ou_all + ou_mean
    ou_all[ou_all < 0] = 0

    # bin rate and generate spikes
    rate_sum = binData(ou_all, [n_trials, n_bin_data]) * dt
    syn_data = gaussian_sp(rate_sum, disp)
    return syn_data, n_bin_data


def twoTauOU_gammaSpikes_withDispersion(theta, dt, binsize, T, n_trials, data_mean, data_var):
    """Generate a two-timescales process with spike counts sampled from a Gamma distribution.
    disperssion parameter (fano factor) of spike generation function is fitted with ABC.

    Parameters
    -----------
    theta : 1d array
        [timescale1, timescale2, coefficient for timescale1, disperssion_parameter].
    dt : float
        temporal resolution for the OU process generation. dt <= binsize.
    binsize : float
        bin-size for binning data and computing the autocorrelation
        (should be the same as observed data and a multiple of dt).
    T : float
        duration of trials.
    n_trials : float
        number of trials.
    data_mean : float
        mean of the spike counts, e.g., is computed from spike counts in the unit of binsize.
    data_var : float
        variance of the spike counts, e.g., is computed from spike counts in the unit of binsize.

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (n_trials * int(T/binsize)).
    n_bin_data : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau1 = np.array(theta[0])
    tau2 = np.array(theta[1])
    coeff = np.array(theta[2])
    disp = np.array(theta[3])

    # setting the params of OU
    v = 1
    D1 = v / tau1
    D2 = v / tau2
    ou_std = np.sqrt(data_var - disp * data_mean) / binsize  # law of total variance
    ou_mean = data_mean / binsize  # law of total expectation
    binned_data = np.arange(0, T + binsize, binsize)
    n_bin_data = len(binned_data) - 1

    # generate OU
    ou_all1 = OU_gen(tau1, D1, dt, T, n_trials)
    ou_all2 = OU_gen(tau2, D2, dt, T, n_trials)
    ou_all = np.sqrt(coeff) * ou_all1 + np.sqrt(1 - coeff) * ou_all2
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check > 10**10:  # check for all-nan values
        return np.zeros((n_trials, n_bin_data)), n_bin_data

    # fit mean and var
    ou_all = ou_std * ou_all + ou_mean
    ou_all[ou_all < 0] = 0

    # bin rate and generate spikes
    rate_sum = binData(ou_all, [n_trials, n_bin_data]) * dt
    syn_data = gamma_sp(rate_sum, disp)
    return syn_data, n_bin_data


def twoTauOU_gaussianSpikes_withDispersion(theta, dt, binsize, T, n_trials, data_mean, data_var):
    """Generate a two-timescales process with spike counts sampled from a Gamma distribution.
    disperssion parameter (fano factor) of spike generation function is fitted with ABC.

    Parameters
    -----------
    theta : 1d array
        [timescale1, timescale2, coefficient for timescale1, disperssion_parameter].
    dt : float
        temporal resolution for the OU process generation. dt <= binsize.
    binsize : float
        bin-size for binning data and computing the autocorrelation
        (should be the same as observed data and a multiple of dt).
    T : float
        duration of trials.
    n_trials : float
        number of trials.
    data_mean : float
        mean of the spike counts, e.g., is computed from spike counts in the unit of binsize.
    data_var : float
        variance of the spike counts, e.g., is computed from spike counts in the unit of binsize.

    Returns
    -------
    syn_data : nd array
        array of binned spike-counts (n_trials * int(T/binsize)).
    n_bin_data : int
        number of bins/samples per trial (required for computing autocorrelation).

    """
    # load params
    tau1 = np.array(theta[0])
    tau2 = np.array(theta[1])
    coeff = np.array(theta[2])
    disp = np.array(theta[3])

    # setting the params of OU
    v = 1
    D1 = v / tau1
    D2 = v / tau2
    ou_std = np.sqrt(data_var - disp * data_mean) / binsize  # law of total variance
    ou_mean = data_mean / binsize  # law of total expectation
    binned_data = np.arange(0, T + binsize, binsize)
    n_bin_data = len(binned_data) - 1

    # generate OU
    ou_all1 = OU_gen(tau1, D1, dt, T, n_trials)
    ou_all2 = OU_gen(tau2, D2, dt, T, n_trials)
    ou_all = np.sqrt(coeff) * ou_all1 + np.sqrt(1 - coeff) * ou_all2
    ou_check = np.max(ou_all)
    if not np.isfinite(ou_check) or ou_check > 10**10:  # check for all-nan values
        return np.zeros((n_trials, n_bin_data)), n_bin_data

    # fit mean and var
    ou_all = ou_std * ou_all + ou_mean
    ou_all[ou_all < 0] = 0

    # bin rate and generate spikes
    rate_sum = binData(ou_all, [n_trials, n_bin_data]) * dt
    syn_data = gaussian_sp(rate_sum, disp)
    return syn_data, n_bin_data


# XXX This entire case is unused in any of the examples
# def oneTauOU_oneF(theta, dt, binsize, T, n_trials, data_mean, data_var):
#     """Generate a one-timescale OU process augmeneted with an additive 1/f spectrum.

#     Parameters
#     -----------
#     theta : 1d array
#         [timescale, 1/f exponent, coefficient for timescale].
#     dt : float
#         temporal resolution for OU process generation.
#     binsize : float
#         bin-size for binning data and computing the autocorrelation.
#     T : float
#         duration of trials.
#     n_trials : float
#         number of trials.
#     data_mean : float
#         mean value of the OU process (average of firing rate).
#     data_var : float
#         variance of the OU process (variance of firing rate).

#     Returns
#     -------
#     syn_data : nd array
#         array of binned spike-counts (n_trials * int(T/binsize)).
#     n_bin_data : int
#         number of bins/samples per trial (required for computing autocorrelation).

#     """
#     # load parameters
#     tau = np.array(theta[0])
#     expon = np.array(theta[1])
#     coeff = np.array(theta[2])

#     # setting params for 1/f
#     fs = T/dt
#     fmax = fs/2
#     deltaF = fmax/(fs)

#     # generate 1/f
#     f_range = np.arange(1,fmax + 1, deltaF)
#     psd = 1/((f_range)**expon)

#     onef = psd_to_timeseries(psd, n_trials)
#     onef = stats.zscore(onef, axis = 1)

#     # setting params for OU
#     binsize = dt
#     v = 1
#     D = v/tau
#     binned_data =  np.arange(0, T + binsize, binsize)
#     n_bin_data = len(binned_data)-1
#     # generate OU
#     ou_all = OU_gen(tau,D,dt,T,n_trials)

#     ou_all = np.sqrt(coeff) * ou_all + np.sqrt(1 - coeff) * onef
#     ou_check = np.max(ou_all)
#     if not np.isfinite(ou_check) or ou_check>10**10: # check for all-nan values
#         return np.zeros((n_trials,n_bin_data)), n_bin_data

#     # fit mean and var
#     ou_std = np.sqrt(data_var)
#     ou_mean = data_mean
#     ou_all = ou_std * ou_all + ou_mean

#     # bin rate
#     syn_data = binData(ou_all, [n_trials,n_bin_data]) * dt
#     return syn_data, n_bin_data
