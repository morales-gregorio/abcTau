"""
Module for fitting the aABC algorithm.
"""

import numpy as np
from simple_abc import pmc_abc


def fit_withABC(
    my_model,
    data_ac,
    prior_dist,
    inter_save_direc,
    inter_filename,
    datasave_path,
    filename_save,
    epsilon_0,
    min_samples,
    steps,
    min_acc_rate,
    parallel=False,
    n_procs=None,
    # disp=None, # This kwarg does nothing in this script!
    resume=None,
    case="general",
):
    """Fits data autocorrelation with a given generative model and saves the results.

    Parameters
    -----------
    my_model : object
        Model object containing the generative model and distance functions
        (check example scripts or tutorials).
    data_ac : 1d array
        Prior distributions for aABC fitting (check example scripts or tutorials).
    prior_dist : list object
        bin-size for binning data and computing the autocorrelation.
    inter_save_direc : string
        directory for saving intermediate results after running each step.
    inter_filename : string
        filename for saving intermediate results after running each step.
    filename_save : string
        filename for saving final results, number of steps and maximumTimeLag
        will be attached to it.
    epsilon_0 : float
        initial error threshold
    min_samples : int
        number of accepted samples in postrior distribution for each step of the aABC.
    steps : int
        maximum number of steps (iterations) for running the aABC algorithm.
    min_acc_rate : float
        minimum proportion of samples accepted in each step, between 0 and 1.
    parallel : boolean, default False
        if run parallel processing.
    n_procs : int, optional, default None
        number of cores used for parallel processing.
    disp : float, default None
        The value of dispersion parameter if computed with the grid search method.
        XXX Does not do anything in this script!!!
    resume : numpy record array, optional
        A record array of a previous pmc sequence to continue the sequence on.

    Returns
    -------
    abc_results : object
        A record containing all aABC output from all steps, including 'theta accepted', 'epsilon'.
    final_step : int
        Last step of running the aABC algorithm.

    """
    # Initialize model object
    model = my_model()
    model.set_prior(prior_dist)

    # give the model our observed data
    model.set_data(data_ac)
    data = data_ac
    # np.warnings.filterwarnings('ignore')

    # fit the model
    abc_results = pmc_abc(
        model,
        data,
        inter_save_direc,
        inter_filename,
        epsilon_0=epsilon_0,
        min_samples=min_samples,
        steps=steps,
        parallel=parallel,
        n_procs=n_procs,
        min_acc_rate=min_acc_rate,
        resume=resume,
        case=case,
    )

    # finding the final step and save the results
    final_step = steps
    for i, result in enumerate(abc_results):
        if result[-1] is None:
            final_step = i
            break
    filename_save = filename_save + "_steps" + str(final_step)
    np.save(datasave_path + filename_save, abc_results)

    print("END OF FITTING!!!")
    print("***********************")

    return abc_results, final_step
