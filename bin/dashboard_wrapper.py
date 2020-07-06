import sys
import os
import numpy as np
import json
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sample_config_distributions(config, num_single_runs):
    # Set a new sampled value from the config distribution as the mean value of the config for a certain number of runs

    if config['N']['type'] == 'normal':
        updated_N = np.random.normal(config['N']['mean'], config['N']['stddev'], num_single_runs)
    else:
        updated_N = np.random.uniform(config['N']['min'], config['N']['max'], num_single_runs)

    if config['R0']['type'] == 'normal':
        updated_R0 = np.random.normal(config['R0']['mean'], config['R0']['stddev'], num_single_runs)
    else:
        updated_R0 = np.random.uniform(config['R0']['min'], config['R0']['max'], num_single_runs)

    # Loop through the alpha values to sample each one
    updated_alpha = np.zeros((num_single_runs, len(config['alpha']), 2))
    for i in range(0, len(config['alpha'])):
        if config['alpha_normal']:
            updated_alpha[:, i, :] = np.array([
                np.random.normal(config['alpha'][i][0], config['alpha'][i][1], num_single_runs),
                [config['alpha'][i][1]] * num_single_runs]).T
        else:
            updated_alpha[:, i, :] = np.array([
                np.random.uniform(config['alpha'][i][0], config['alpha'][i][1], num_single_runs),
                [config['alpha'][i][1]] * num_single_runs]).T

    if config['delayHOS']['type'] == 'normal':
        updated_delayHOS = np.random.normal(config['delayHOS']['mean'], config['delayHOS']['stddev'], num_single_runs)
    else:
        updated_delayHOS = np.random.uniform(config['delayHOS']['min'], config['delayHOS']['max'], num_single_runs)

    if config['delayHOSREC']['type'] == 'normal':
        updated_delayHOSREC = np.random.normal(config['delayHOSREC']['mean'], config['delayHOSREC']['stddev'],
                                               num_single_runs)
    else:
        updated_delayHOSREC = np.random.uniform(config['delayHOSREC']['min'], config['delayHOSREC']['max'],
                                               num_single_runs)

    if config['delayHOSD']['type'] == 'normal':
        updated_delayHOSD = np.random.normal(config['delayHOSD']['mean'], config['delayHOSD']['stddev'],
                                             num_single_runs)
    else:
        updated_delayHOSD = np.random.uniform(config['delayHOSD']['min'], config['delayHOSD']['max'], num_single_runs)

    if config['delayICUCAND']['type'] == 'normal':
        updated_delayICUCAND = np.random.normal(config['delayICUCAND']['mean'], config['delayICUCAND']['stddev'],
                                                num_single_runs)
    else:
        updated_delayICUCAND = np.random.uniform(config['delayICUCAND']['min'], config['delayICUCAND']['max'],
                                                num_single_runs)

    if config['delayICUD']['type'] == 'normal':
        updated_delayICUD = np.random.normal(config['delayICUD']['mean'], config['delayICUD']['stddev'],
                                             num_single_runs)
    else:
        updated_delayICUD = np.random.uniform(config['delayICUD']['min'], config['delayICUD']['max'], num_single_runs)

    if config['delayICUREC']['type'] == 'normal':
        updated_delayICUREC = np.random.normal(config['delayICUREC']['mean'], config['delayICUREC']['stddev'],
                                             num_single_runs)
    else:
        updated_delayICUREC = np.random.uniform(config['delayICUREC']['min'], config['delayICUREC']['max'],
                                               num_single_runs)

    if config['hosfrac']['type'] == 'normal':
        updated_hosfrac = np.random.normal(config['hosfrac']['mean'], config['hosfrac']['stddev'], num_single_runs)
    else:
        updated_hosfrac = np.random.uniform(config['hosfrac']['min'], config['hosfrac']['max'], num_single_runs)

    if config['dfrac']['type'] == 'normal':
        updated_dfrac = np.random.normal(config['dfrac']['mean'], config['dfrac']['stddev'], num_single_runs)
    else:
        updated_dfrac = np.random.uniform(config['dfrac']['min'], config['dfrac']['max'], num_single_runs)

    if config['icudfrac']['type'] == 'normal':
        updated_icudfrac = np.random.normal(config['icudfrac']['mean'], config['icudfrac']['stddev'], num_single_runs)
    else:
        updated_icudfrac = np.random.uniform(config['icudfrac']['min'], config['icudfrac']['max'], num_single_runs)

    if config['icufracscale']['type'] == 'normal':
        updated_icufracscale = np.random.normal(config['icufracscale']['mean'], config['icufracscale']['stddev'],
                                                num_single_runs)
    else:
        updated_icufracscale = np.random.uniform(config['icufracscale']['min'], config['icufracscale']['max'],
                                                num_single_runs)

    # Create the return data as a list of configs
    updated_config = []
    for i in range(0, num_single_runs):
        updated_config.append(copy.deepcopy(config))

        if config['N']['type'] == 'normal':
            updated_config[i]['N']['mean'] = updated_N[i]

        if config['R0']['type'] == 'normal':
            updated_config[i]['R0']['mean'] = updated_R0[i]

        updated_config[i]['alpha'] = updated_alpha[i].tolist()

        if config['delayHOS']['type'] == 'normal':
            updated_config[i]['delayHOS']['mean'] = updated_delayHOS[i]

        if config['delayHOSREC']['type'] == 'normal':
            updated_config[i]['delayHOSREC']['mean'] = updated_delayHOSREC[i]

        if config['delayHOSD']['type'] == 'normal':
            updated_config[i]['delayHOSD']['mean'] = updated_delayHOSD[i]

        if config['delayICUCAND']['type'] == 'normal':
            updated_config[i]['delayICUCAND']['mean'] = updated_delayICUCAND[i]

        if config['delayICUD']['type'] == 'normal':
            updated_config[i]['delayICUD']['mean'] = updated_delayICUD[i]

        if config['delayICUREC']['type'] == 'normal':
            updated_config[i]['delayICUREC']['mean'] = updated_delayICUREC[i]

        if config['hosfrac']['type'] == 'normal':
            updated_config[i]['hosfrac']['mean'] = updated_hosfrac[i]

        if config['dfrac']['type'] == 'normal':
            updated_config[i]['dfrac']['mean'] = updated_dfrac[i]

        if config['icudfrac']['type'] == 'normal':
            updated_config[i]['icudfrac']['mean'] = updated_icudfrac[i]

        if config['icufracscale']['type'] == 'normal':
            updated_config[i]['icufracscale']['mean'] = updated_icufracscale[i]

    return updated_config


def generate_updated_config(updated_config, results):
    try:
        n_index = results['mnames'].tolist().index('n')
        updated_config['N']['type'] = 'normal'
        updated_config['N']['mean'] = results['mvalues'][n_index][2]
        updated_config['N']['stddev'] = results['mvalues'][n_index][3]
    except ValueError:
        pass

    try:
        r0_index = results['mnames'].tolist().index('r0')
        updated_config['R0']['type'] = 'normal'
        updated_config['R0']['mean'] = results['mvalues'][r0_index][2]
        updated_config['R0']['stddev'] = results['mvalues'][r0_index][3]
    except ValueError:
        pass

    try:
        alpha_index = results['mnames'].tolist().index('alpha')
        updated_config['alpha_normal'] = True
        updated_config['alpha'] = [[results['mvalues'][alpha_index + ind][2], results['mvalues'][alpha_index + ind][3]]
                                   for ind, x in enumerate(updated_config['alpha'])]
    except ValueError:
        pass

    try:
        delayHOS_index = results['mnames'].tolist().index('delay_hos')
        updated_config['delayHOS']['type'] = 'normal'
        updated_config['delayHOS']['mean'] = results['mvalues'][delayHOS_index][2]
        updated_config['delayHOS']['stddev'] = results['mvalues'][delayHOS_index][3]
    except ValueError:
        pass

    try:
        delayHOSD_index = results['mnames'].tolist().index('delay_hosd')
        updated_config['delayHOSD']['type'] = 'normal'
        updated_config['delayHOSD']['mean'] = results['mvalues'][delayHOSD_index][2]
        updated_config['delayHOSD']['stddev'] = results['mvalues'][delayHOSD_index][3]
    except ValueError:
        pass

    try:
        delayICUD_index = results['mnames'].tolist().index('delay_icud')
        updated_config['delayICUD']['type'] = 'normal'
        updated_config['delayICUD']['mean'] = results['mvalues'][delayICUD_index][2]
        updated_config['delayICUD']['stddev'] = results['mvalues'][delayICUD_index][3]
    except ValueError:
        pass

    try:
        delayICUREC_index = results['mnames'].tolist().index('delay_icurec')
        updated_config['delayICUREC']['type'] = 'normal'
        updated_config['delayICUREC']['mean'] = results['mvalues'][delayICUREC_index][2]
        updated_config['delayICUREC']['stddev'] = results['mvalues'][delayICUREC_index][3]
    except ValueError:
        pass

    try:
        hosfrac_index = results['mnames'].tolist().index('hosfrac')
        updated_config['hosfrac']['type'] = 'normal'
        updated_config['hosfrac']['mean'] = results['mvalues'][hosfrac_index][2]
        updated_config['hosfrac']['stddev'] = results['mvalues'][hosfrac_index][3]
    except ValueError:
        pass

    try:
        dfrac_index = results['mnames'].tolist().index('dfrac')
        updated_config['dfrac']['type'] = 'normal'
        updated_config['dfrac']['mean'] = results['mvalues'][dfrac_index][2]
        updated_config['dfrac']['stddev'] = results['mvalues'][dfrac_index][3]
    except ValueError:
        pass

    try:
        icudfrac_index = results['mnames'].tolist().index('icudfrac')
        updated_config['icudfrac']['type'] = 'normal'
        updated_config['icudfrac']['mean'] = results['mvalues'][icudfrac_index][2]
        updated_config['icudfrac']['stddev'] = results['mvalues'][icudfrac_index][3]
    except ValueError:
        pass

    try:
        delayICUD_sd_index = results['mnames'].tolist().index('icud_sd')
        # Gauss. smooth dist. delayICUD mean
        updated_config['delayICUD']['smooth_sd'] = results['mvalues'][delayICUD_sd_index][2]
        # Gauss. smooth dist. delayICUD stddev
        updated_config['delayICUD']['smooth_sd_sd'] = results['mvalues'][delayICUD_sd_index][3]
    except ValueError:
        pass

    try:
        delayHOSD_sd_index = results['mnames'].tolist().index('hosd_sd')
        # Gauss. smooth dist. delayHOSD mean
        updated_config['delayHOSD']['smooth_sd'] = results['mvalues'][delayHOSD_sd_index][2]
        # Gauss. smooth dist. delayHOSD stddev
        updated_config['delayHOSD']['smooth_sd_sd'] = results['mvalues'][delayHOSD_sd_index][3]
    except ValueError:
        pass

    try:
        icufracscale_index = results['mnames'].tolist().index('icufracscale')
        updated_config['icufracscale']['type'] = 'normal'
        updated_config['icufracscale']['mean'] = results['mvalues'][icufracscale_index][2]
        updated_config['icufracscale']['stddev'] = results['mvalues'][icufracscale_index][3]
    except ValueError:
        pass

    return updated_config


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
        # try:
        #     num_single_runs = config['num_single_runs'] - 1
        # except KeyError:
        #     num_single_runs = 4

        # Run the single run with the mean provided
        results_mean = single_run_mean(config, data)

        # # Run the single run for the remaining times with values sampled from the config distributions
        # results_sampled = np.zeros((num_single_runs, results_mean.shape[0], results_mean.shape[1]))
        # # Get a list of configs with the mean changed to be a sampled value of the original config mean and standard
        # # deviation.
        # sampled_configs = sample_config_distributions(config, num_single_runs)
        # for i in range(0, num_single_runs):
        #     results_sampled[i] = single_run_mean(sampled_configs[i], data)

        # Rearrange the output data starting from day 1
        [start_index] = np.where(results_mean[0] == 1)[0]
        # Set the results
        dashboard_data = {
            'susceptible': np.concatenate((results_mean[0, start_index:, None],
                                           results_mean[1, start_index:, None]), axis=1),
                                           # results_sampled[:, 1, start_index:].T), axis=1),
            'exposed': np.concatenate((results_mean[0, start_index:, None],
                                       results_mean[2, start_index:, None]), axis=1),
                                       # results_sampled[:, 2, start_index:].T), axis=1),
            'infected': np.concatenate((results_mean[0, start_index:, None],
                                        results_mean[3, start_index:, None]), axis=1),
                                        # results_sampled[:, 3, start_index:].T), axis=1),
            'removed': np.concatenate((results_mean[0, start_index:, None],
                                       results_mean[4, start_index:, None]), axis=1),
                                       # results_sampled[:, 4, start_index:].T), axis=1),
            'hospitalized': np.concatenate((results_mean[0, start_index:, None],
                                            results_mean[5, start_index:, None]), axis=1),
                                            # results_sampled[:, 5, start_index:].T), axis=1),
            'hospitalizedcum': np.concatenate((results_mean[0, start_index:, None],
                                               results_mean[6, start_index:, None]), axis=1),
                                               # results_sampled[:, 6, start_index:].T), axis=1),
            'ICU': np.concatenate((results_mean[0, start_index:, None],
                                   results_mean[7, start_index:, None]), axis=1),
                                   # results_sampled[:, 7, start_index:].T), axis=1),
            'icu_cum': np.concatenate((results_mean[0, start_index:, None],
                                       results_mean[8, start_index:, None]), axis=1),
                                       # results_sampled[:, 8, start_index:].T), axis=1),
            'recovered': np.concatenate((results_mean[0, start_index:, None],
                                         results_mean[9, start_index:, None]), axis=1),
                                         # results_sampled[:, 9, start_index:].T), axis=1),
            'dead': np.concatenate((results_mean[0, start_index:, None],
                                    results_mean[10, start_index:, None]), axis=1),
                                    # results_sampled[:, 10, start_index:].T), axis=1),
            'infected_cum': np.concatenate((results_mean[0, start_index:, None],
                                            results_mean[11, start_index:, None]), axis=1),
                                            # results_sampled[:, 11, start_index:].T), axis=1),
            'alpha': np.concatenate((results_mean[0, start_index:, None],
                                     results_mean[12, start_index:, None]), axis=1)}
                                     # results_sampled[:, 12, start_index:].T), axis=1)}
    else:
        # Make the input data and IC files
        get_and_save_nl_data(config['startdate'], config['country'], config['icdatafile'])
        # Make the IC fraction file
        save_and_plot_IC(config, config['output_base_filename'], 0)

        # Run the model
        dashboard_data, results = run_esmda_model(config['output_base_filename'], config, data,
                                                  save_plots=0, save_files=0)

        # Update the config with the new posteriors to return
        updated_config = generate_updated_config(updated_config, results)

    # Return the result data (for both multiple or single run) as well as the updated config file. The dashboard_data
    # contains the following columns for the respective variables:
    # ~~ Full run ~~
    #   - alpha:
    #     - time | P5 | P25 | P50 | P75 | P95
    #   - alpharun, dead, hospitalized, hospitalized cumulative, icu, infected:
    #     - time | mean | P5 | P30 | P50 | P70 | P95 | observed
    # ~~ Single run ~~
    #   - susceptible, exposed, infected, removed, hospitalized, hospitalizedcum,
    #     ICU, icu_cum, recovered, dead, infected_cum, alpha:
    #     - time | single run with original updated config | remaining number of single runs with sampled config
    return dashboard_data, updated_config


if __name__ == '__main__':
    # Input parameters:
    #   - config: json format with all the input variables
    run_dashboard_wrapper(sys.argv[1])
