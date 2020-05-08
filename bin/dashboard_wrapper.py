import sys
import os
import numpy as np
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_dashboard_wrapper(config_input):
    # Input is a json with all the input data
    # Extra keys that are NOT in the standard config files are:
    #   - output_base_filename: the base name for the icu frac file to be saved, e.g. 'netherlands_dashboard'
    #   - single_run: a boolean if it should only perform a single run

    # Relative imports
    from src.io_func import load_data
    from bin.corona_esmda import run_esmda_model, single_run_mean
    from bin.analyseIC import save_and_plot_IC
    from src.api_nl_data import get_and_save_nl_data

    # Load the model configuration and the data (observed cases)
    config = json.loads(config_input)
    data = load_data(config)

    updated_config = config
    if config['single_run']:
        # Run the single run model
        single_run_results = single_run_mean(config, data)

        # Rearrange the output data starting from day 1
        [start_index] = np.where(single_run_results[0] == 1)[0]
        dashboard_data = {
            'susceptible':      np.concatenate((single_run_results[0][start_index:, None],
                                                single_run_results[1][start_index:, None]), axis=1),
            'exposed':          np.concatenate((single_run_results[0][start_index:, None],
                                                single_run_results[2][start_index:, None]), axis=1),
            'infected':         np.concatenate((single_run_results[0][start_index:, None],
                                                single_run_results[3][start_index:, None]), axis=1),
            'removed':          np.concatenate((single_run_results[0][start_index:, None],
                                                single_run_results[4][start_index:, None]), axis=1),
            'hospitalized':     np.concatenate((single_run_results[0][start_index:, None],
                                                single_run_results[5][start_index:, None]), axis=1),
            'hospitalizedcum':  np.concatenate((single_run_results[0][start_index:, None],
                                                single_run_results[6][start_index:, None]), axis=1),
            'ICU':              np.concatenate((single_run_results[0][start_index:, None],
                                                single_run_results[7][start_index:, None]), axis=1),
            'icu_cum':          np.concatenate((single_run_results[0][start_index:, None],
                                                single_run_results[8][start_index:, None]), axis=1),
            'recovered':        np.concatenate((single_run_results[0][start_index:, None],
                                                single_run_results[9][start_index:, None]), axis=1),
            'dead':             np.concatenate((single_run_results[0][start_index:, None],
                                                single_run_results[10][start_index:, None]), axis=1),
            'infected_cum':     np.concatenate((single_run_results[0][start_index:, None],
                                                single_run_results[11][start_index:, None]), axis=1),
            'alpha':            np.concatenate((single_run_results[0][start_index:, None],
                                                single_run_results[12][start_index:, None]), axis=1),
        }
    else:
        # Make the input data and IC files
        get_and_save_nl_data(config['startdate'], config['country'], config['icdatafile'])
        # Make the IC fraction file
        save_and_plot_IC(config, config['output_base_filename'], 0)

        # Run the model
        dashboard_data, results = run_esmda_model(config['output_base_filename'], config, data,
                                                  save_plots=0, save_files=0)

        # Update the config with the new posteriors to return
        updated_config['N']['type'] = 'normal'
        updated_config['N']['mean'] = results['mvalues'][0][2]
        updated_config['N']['stddev'] = results['mvalues'][0][3]
        updated_config['R0']['mean'] = results['mvalues'][1][2]
        updated_config['R0']['stddev'] = results['mvalues'][1][3]
        updated_config['delayHOSD']['mean'] = results['mvalues'][-7][2]
        updated_config['delayHOSD']['stddev'] = results['mvalues'][-7][3]
        updated_config['delayICUD']['mean'] = results['mvalues'][-6][2]
        updated_config['delayICUD']['stddev'] = results['mvalues'][-6][3]
        updated_config['delayICUREC']['mean'] = results['mvalues'][-5][2]
        updated_config['delayICUREC']['stddev'] = results['mvalues'][-5][3]
        updated_config['hosfrac']['mean'] = results['mvalues'][-4][2]
        updated_config['hosfrac']['stddev'] = results['mvalues'][-4][3]
        updated_config['dfrac']['mean'] = results['mvalues'][-3][2]
        updated_config['dfrac']['stddev'] = results['mvalues'][-3][3]
        updated_config['delayHOSD']['smooth_sd'] = results['mvalues'][-2][2]  # Gauss. smooth dist. delayHOSD mean
        updated_config['delayHOSD']['smooth_sd_sd'] = results['mvalues'][-2][3]  # Gauss. smooth dist. delayHOSD stddev
        updated_config['icufracscale']['mean'] = results['mvalues'][-1][2]
        updated_config['icufracscale']['stddev'] = results['mvalues'][-1][3]

    # Return the result data (for both multiple or single run) as well as the updated config file
    return dashboard_data, updated_config


if __name__ == '__main__':
    # Input parameters:
    #   - config: json format with all the input variables
    run_dashboard_wrapper(sys.argv[1])
