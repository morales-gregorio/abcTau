"""
Module containing different metric for computing summay statistics
(autocrrelation or power spectrum).
"""

import numpy as np


def comp_summary(data, summary_metric, normalize, dt, binsize, T, n_bins, max_lag=None):
    """
    Compute summary statistics for a given metric.

    Parameters
    -----------
    data : nd array
        time-series from binned data (n_trials * n_bins).
    summary_metric : string
        metric for computing summay statistics ('comp_cc', 'comp_ac_fft', 'comp_psd').
    normalize : string
        if normalize the autocorrelation to the zero time-lag or PSD to have integral equal to 1.
    dt : float
        temporal resolution of data (or binsize of spike counts).
    max_lag : float
        maximum time-lag for computing cross- or auto-correlation.
    binsize : float
        bin-size used for binning data.
    T : float
        duration of each trial/time-series.
    n_bins : int
        number of time-bins in each trial of binned_data1 and x2.
    max_lag : float, default None
        maximum time-lag for computing autocorrelation.

    Returns
    -------
    ach : 1d array
        average non-normalized cross- or auto-correlation across all trials.
    """
    if summary_metric == "comp_cc":
        # compute ach time domain
        summary = comp_cc(data, data, max_lag, binsize, n_bins)
        if normalize:
            # normalize autocorrelation (optional depending on the application)
            summary = summary / summary[0]
    elif summary_metric == "comp_ac_fft":
        # compute ach fft
        summary = comp_ac_fft(data)
        max_lag_bin = round(max_lag / binsize)
        summary = summary[0:max_lag_bin]
        if normalize:
            # normalize autocorrelation (optional depending on the application)
            summary = summary / summary[0]
    elif summary_metric == "comp_psd":
        # compute psd
        summary = comp_psd(data, T, dt)
        if normalize:
            # normalize the psd to not to deal with large numbers
            summary = summary / np.sum(summary)
    else:
        summary = None
        raise ValueError("unknown summary statistics")

    return summary


def comp_cc_2(binned_data1, x2, max_lag, binsize, n_bins):
    """Compute cross- or auto-correlation from binned data (without normalization).

    Parameters
    -----------
    binned_data1, x2 : nd array
        time-series from binned data (n_trials * n_bins).
    D : float
        diffusion parameter.
    max_lag : float
        maximum time-lag for computing cross- or auto-correlation.
    binsize : float
        bin-size used for binning binned_data1 and x2.
    n_bins : int
        number of time-bins in each trial of binned_data1 and x2.


    Returns
    -------
    ach : 1d array
        average non-normalized cross- or auto-correlation across all trials.
    """
    n_trials1 = np.shape(binned_data1)[0]
    n_trials2 = np.shape(x2)[0]

    if n_trials1 != n_trials2:
        raise ValueError("n_trials1 != n_trials2")

    n_lag_bins = int(np.ceil((max_lag) / binsize) + 1)
    ach = np.zeros((n_lag_bins))
    for trial in range(n_trials1):
        xt1 = binned_data1[trial]
        xt2 = x2[trial]
        for lag_idx in range(0, n_lag_bins):
            # index to take this part of the array 1
            ind1 = np.arange(np.max([0, -lag_idx]), np.min([n_bins - lag_idx, n_bins]))

            # index to take this part of the array 2
            ind2 = np.arange(np.max([0, lag_idx]), np.min([n_bins + lag_idx, n_bins]))

            ach[lag_idx] = ach[lag_idx] + (
                np.dot(xt1[ind1], xt2[ind2]) / (len(ind1)) - np.mean(xt1[ind1]) * np.mean(xt2[ind2])
            )

    return ach / n_trials1


def comp_cc(binned_data1, x2, max_lag, binsize, n_bins):
    """Compute cross- or auto-correlation from binned data (without normalization).
    Uses matrix computations to speed up, preferred when multiple processes are available.

    Parameters
    -----------
    binned_data1, x2 : nd array
        time-series from binned data (n_trials * n_bins).
    D : float
        diffusion parameter.
    max_lag : float
        maximum time-lag for computing cross- or auto-correlation.
    binsize : float
        bin-size used for binning binned_data1 and x2.
    n_bins : int
        number of time-bins in each trial of binned_data1 and x2.

    Returns
    -------
    ach : 1d array
        average non-normalized cross- or auto-correlation across all trials.
    """
    n_lag_bins = int(np.ceil((max_lag) / binsize) + 1) - 1
    ach = np.zeros((n_lag_bins))
    for lag_idx in range(0, n_lag_bins):

        # index to take this part of the array 1
        ind1 = np.arange(np.max([0, -lag_idx]), np.min([n_bins - lag_idx, n_bins]))

        # index to take this part of the array 2
        ind2 = np.arange(np.max([0, lag_idx]), np.min([n_bins + lag_idx, n_bins]))

        cov_trs = np.sum((binned_data1[:, ind1] * x2[:, ind2]), axis=1) / len(ind1)
        ach[lag_idx] = np.mean(
            cov_trs - np.mean(binned_data1[:, ind1], axis=1) * np.mean(x2[:, ind2], axis=1)
        )

    return ach


def comp_ac_fft_middlepad(data):
    """Compute auto-correlations from binned data (without normalization).
    Uses FFT after zero-padding the time-series in the middle.

    Parameters
    -----------
    data : nd array
        time-series from binned data (n_trials * n_bins).

    Returns
    -------
    ach : 1d array
        average non-normalized auto-correlation across all trials.
    """
    n_trials = np.shape(data)[0]
    ac_sum = 0
    for trial in range(n_trials):
        x = data[trial]
        xp = np.fft.ifftshift((x - np.average(x)))
        (n,) = xp.shape
        xp = np.r_[xp[: n // 2], np.zeros_like(xp), xp[n // 2 :]]
        f = np.fft.fft(xp)
        p = np.absolute(f) ** 2
        pi = np.fft.ifft(p)
        ach = np.real(pi)[: n - 1] / np.arange(1, n)[::-1]
        ac_sum = ac_sum + ach
    ach = ac_sum / n_trials
    return ach


def comp_ac_fft_middlepad_zscore(data):
    """Compute auto-correlations from binned data (without normalization).
    Uses FFT after z-scoring and zero-padding the time-series in the middle.

    Parameters
    -----------
    data : nd array
        time-series from binned data (n_trials * n_bins).

    Returns
    -------
    ach : 1d array
        average non-normalized auto-correlation across all trials.
    """
    n_trials = np.shape(data)[0]
    ac_sum = 0
    for trial in range(n_trials):
        x = data[trial]
        xp = np.fft.ifftshift((x - np.average(x)) / np.std(x))
        (n,) = xp.shape
        xp = np.r_[xp[: n // 2], np.zeros_like(xp), xp[n // 2 :]]
        f = np.fft.fft(xp)
        p = np.absolute(f) ** 2
        pi = np.fft.ifft(p)
        ach = np.real(pi)[: n - 1] / np.arange(1, n)[::-1]
        ac_sum = ac_sum + ach
    ach = ac_sum / n_trials
    return ach


def comp_ac_fft(data):
    """Compute auto-correlations from binned data (without normalization).
    Uses FFT after zero-padding the time-series in the right side.

    Parameters
    -----------
    data : nd array
        time-series from binned data (n_trials * n_bins).

    Returns
    -------
    ach : 1d array
        average non-normalized auto-correlation across all trials.
    """
    n = np.shape(data)[1]
    xp = data - data.mean(1)[:, None]
    xp = np.concatenate((xp, np.zeros_like(xp)), axis=1)
    f = np.fft.fft(xp)
    p = np.absolute(f) ** 2
    pi = np.fft.ifft(p)
    ac_all = np.real(pi)[:, : n - 1] / np.arange(1, n)[::-1]
    ach = np.mean(ac_all, axis=0)
    return ach


def comp_psd(x, T, dt):
    """Compute the power spectrum density (PSD) using a Hamming window and direct fft.

    Parameters
    -----------
    binned_data1 : nd array
        time-series from binned data (n_trials * n_bins).
    T : float
        duration of each trial/time-series.
    dt : float
        temporal resolution of data (or binsize of spike counts).


    Returns
    -------
    psd : 1d array
        average  power spectrum density (PSD) across all trials.
    """
    # fs = T/dt
    n_points = len(x[0])
    x_windowed = (x - x.mean(1)[:, None]) * np.hamming(n_points)
    PSD = np.mean(np.abs(np.fft.rfft(x_windowed)) ** 2, axis=0)[1:-1]

    return PSD
