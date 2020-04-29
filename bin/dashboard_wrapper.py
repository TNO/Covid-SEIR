import sys
import os
from pathlib import Path
import numpy as np
from src.io_func import load_data, load_config, save_results, add_hammer_to_results
from bin.corona_esmda import I_TIME, get_calibration_modes, get_calibration_errors, s_calmodes, i_calmodes, \
    o_calmodes, map_model_to_obs, apply_hammer, save_posterior_and_prior, save_prior_and_posterior_alpha, \
    S_HOS, S_ICU, S_HOSCUM, S_DEAD, S_INF, S_ALPHARUN, \
    I_HOS, I_ICU, I_HOSCUM, I_DEAD, I_INF, \
    O_HOS, O_ICU, O_HOSCUM, O_DEAD, O_CUMINF, O_ALPHARUN
from src.tools import generate_hos_actual, generate_zero_columns
from src.esmda import InversionMethods
from src.coronaSEIR import base_seir_model
from src.parse import parse_config, reshape_prior, get_mean
from bin.confidencecurves import plot_confidence
from bin.preprocess import preprocess
import json


def run_esmda(config_data):
    # Input parameters:
    #   - config_data: json format with all the input variables

    # Load the model configuration and the data (observed cases)
    if config_data == 'null':
        configpath = '../configs/netherlands_april25.json'
        config = load_config(configpath)
    else:
        config = json.loads(config_data)

    data = load_data(config)
    # Concatenate additional column for actual observed hospitalized based
    t_obs = data[:, I_TIME]

    useworldfile = config['worldfile']
    if not useworldfile:
        data = generate_hos_actual(data, config)
    else:
        data = generate_zero_columns(data, config)
        # Run the forward model to obtain a prior ensemble of models
    # save_input_data(configpath, data)

    calibration_modes = get_calibration_modes(config['calibration_mode'])
    observation_errors = get_calibration_errors(config['observation_error'])
    calibration_mode = calibration_modes[0]

    found = False
    for i, calmode in enumerate(s_calmodes):
        if calmode == calibration_mode:
            print('ensemble fit on : ', calibration_modes[0], ' with error ', observation_errors[0])
            y_obs = data[:, i_calmodes[i]]
            output_index = [o_calmodes[i]]
            error = np.ones_like(y_obs) * observation_errors[0]
            found = True
    if not found:
        raise NotImplementedError

    for ical, calibration_mode in enumerate(calibration_modes):
        if ical > 0:
            print('additional ensemble fit on : ', calibration_modes[ical], ' with error ', observation_errors[ical])
            for i, calmode in enumerate(s_calmodes):
                if calmode == calibration_mode:
                    y_obs2 = data[:, i_calmodes[i]]
                    output_index2 = [o_calmodes[i]]
                    error2 = np.ones_like(y_obs2) * observation_errors[ical]
                    y_obs = np.append(y_obs, y_obs2)
                    # output_index = [output_index[0], O_ICU]
                    output_index = np.append(output_index, output_index2)
                    error = np.append(error, error2)
                    found = True
            if not found:
                raise NotImplementedError

    # Load inversion method and define forward model
    try:
        ni = config['esmda_iterations']
    except KeyError:
        ni = 8
    im = InversionMethods(ni=ni)
    forward = base_seir_model
    np.random.seed(12345)

    # Parse the configuration json
    ndata = np.size(data[:, 0])
    m_prior, fwd_args = parse_config(config, ndata)
    m_prior = reshape_prior(m_prior)

    # Estimated uncertainty on the data
    # sigma = np.sqrt(y_obs).clip(min=10)

    # error = config['observation_error']
    # if calibration_mode == S_HOS or calibration_mode == S_HOSCUM:
    #    error = 2 * error
    # sigma = error * np.ones_like(y_obs)

    sigma = error

    add_arg = [fwd_args]
    kwargs = {'t_obs': t_obs,
              'output_index': output_index,
              'G': map_model_to_obs,
              'print_results': 1}

    results = im.es_mda(forward, m_prior, y_obs, sigma, *add_arg, **kwargs)

    dashboard_data = save_results_to_csv(results, fwd_args, config, configpath, t_obs,
                                         data, calibration_mode, output_index)
    # base = (os.path.split(configpath)[-1]).split('.')[0]
    # outpath = os.path.join(os.path.split(os.getcwd())[0], 'output', base + '_output.h5')
    # save_results(results, fwd_args, config, outpath, data, mode='esmda')

    # Check if we want to apply a 'hammer'
    try:
        pass
        # hammer_icu = config['hammer_ICU']
        # hammer_slope = config['hammer_slope']
        # hammer_alpha = config['hammer_alpha']
        # assert len(hammer_alpha) == 2
        # time_delay = config['time_delay']
        # hammered_results = apply_hammer(results['fw'][-1], results['M'][-1], fwd_args, hammer_icu, hammer_slope,
        #                                 hammer_alpha, time_delay, data_end=data[-1, 0])
        # # TODO: Add hammer results to h5 file and plotting
        # results['fw'][-1] = hammered_results
        # # plot_prior_and_posterior(results, fwd_args, config, configpath, t_obs, data, calibration_mode, output_index)
        # save_results_to_csv(results, fwd_args, config, configpath, t_obs, data, calibration_mode, output_index)
        # add_hammer_to_results(hammered_results, outpath, mode='esmda')

    except KeyError as e:
        pass  # Don't apply hammer, you're done early!

    return dashboard_data


def save_results_to_csv(results, fwd_args, config, configpath, t_obs, data, calibration_mode, output_index):
    # plot_prior_and_posterior(results, fwd_args, config, configpath, t_obs, data, calibration_mode, output_index)

    base = (os.path.split(configpath)[-1]).split('.')[0]
    subdir = 'output\\dashboard'
    outpath = os.path.join(os.path.split(os.getcwd())[0], subdir, base)
    Path(os.path.join(os.path.split(os.getcwd())[0], subdir)).mkdir(parents=True, exist_ok=True)

    calmodes = [S_HOS, S_ICU, S_HOSCUM, S_DEAD, S_INF, S_ALPHARUN]
    o_indices = [O_HOS, O_ICU, O_HOSCUM, O_DEAD, O_CUMINF, O_ALPHARUN]
    y_obs_s = [data[:, I_HOS], data[:, I_ICU], data[:, I_HOSCUM], data[:, I_DEAD], data[:, I_INF], []]

    posterior = results['fw'][-1]
    time = fwd_args['time'] - fwd_args['time_delay']

    # For saving the csv files
    p_values = config['p_values']
    steps = np.arange(1, time.max() + 1)
    t_ind = [np.where(time == a)[0][0] for a in steps]
    h_pvalues = ['P' + str(int(100 * a)) for a in p_values]
    header = 'time,mean,' + ','.join(h_pvalues) + ',observed'

    dashboard_data = {}

    for i, calmode in enumerate(calmodes):
        output_index = o_indices[i]
        y_obs = y_obs_s[i]

        posterior_curves = np.array([member[output_index, :] for member in posterior]).T
        post_mean = np.mean(posterior_curves, axis=-1)

        dashboard_data[calmode] = save_posterior_and_prior(posterior_curves, t_ind, p_values, y_obs, steps,
                                                           post_mean, outpath, calmode, config, header)

    dashboard_data['alpha'] = save_prior_and_posterior_alpha(results, config, steps, outpath)

    return dashboard_data


def run_conf_curves(config):
    # Input parameters:
    #   - config: json format with all the input variables
    configpath = '../configs/netherlands_april25.json'

    # Load the model configuration and the data (observed cases)
    config = load_config(configpath)
    # config = json.loads(config_data)
    data = load_data(config)

    useworldfile = config['worldfile']
    if (not useworldfile):
        data = generate_hos_actual(data, config)
    else:
        data = generate_zero_columns(data, config)

    base = (os.path.split(configpath)[-1]).split('.')[0]
    outpath = os.path.join(os.path.split(os.getcwd())[0], 'output\\dashboard', base)

    # Generate the confidence curves
    print('Generating confidence curves...')
    plot_confidence(outpath, config, data, config['startdate'])


def run_dashboard_wrapper(config_data):
    preprocess("../configs/netherlands_april25.json", "../res/icdata25nice.txt")
    dashboard_data = run_esmda(config_data)
    # run_conf_curves(config_data)
    return dashboard_data


if __name__ == '__main__':
    # Input parameters:
    #   - config: json format with all the input variables
    run_dashboard_wrapper(sys.argv[1])
