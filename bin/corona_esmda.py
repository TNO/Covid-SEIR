from src.coronaSEIR import base_seir_model
from src.esmda import InversionMethods
from src.io_func import load_data, load_config, hospital_forecast_to_txt, save_results, save_input_data
from src.parse import parse_config, reshape_prior
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from src.tools import generate_hos_actual, generate_zero_columns

I_TIME = 0
I_INF = 1
I_DEAD = 2
I_REC = 3
I_HOSCUM = 4
I_ICU = 5
I_HOS = 6

O_TIME = 0
O_SUS = 1
O_EXP = 2
O_INF = 3
O_REM = 4
O_HOS = 5
O_HOSCUM = 6
O_ICU = 7
O_ICUCUM = 8
O_REC = 9
O_DEAD = 10

S_HOS = 'hospitalized'
S_ICU = 'ICU'
S_HOSCUM = 'hospitalizedcum'
S_DEAD = 'dead'


def map_model_to_obs(fwd, t_obs, output_index):
    t_ind = [np.where(fwd[O_TIME, :] == a)[0][0] for a in t_obs]
    return fwd[output_index, t_ind]





def plot_prior_and_posterior(results, fwd_args, config, configpath, t_obs, data, calibration_mode, output_index):
    # Plot prior
    base = (os.path.split(configpath)[-1]).split('.')[0]
    outpath = os.path.join(os.path.split(os.getcwd())[0], 'output', base)



    calmodes = [S_HOS, S_ICU, S_HOSCUM, S_DEAD]
    o_indices = [O_HOS, O_ICU, O_HOSCUM, O_DEAD]
    #useworldfile = config['worldfile']
    #if (useworldfile):
    #    calmodes = [ S_DEAD]
    #    o_indices = [ O_DEAD]
    #else:
    #    # check if I_HOS is present in the data, if not generate it
    #    print('number of columns in data ', data.shape[1])
    #    print('number of rows in data ', data.shape[0])
    #    ncol = data.shape[1]

    #    data = generate_hos_actual(data, config)


    for i, calmode in enumerate(calmodes):
        output_index = o_indices[i]

        if calmode == S_HOS:
            title = 'Hospitalized'
            ymax = config['YMAX'] * config['hosfrac'] * 2
            y_obs = data[:, I_HOS]
        elif calmode == S_ICU:
            title = 'ICU'
            ymax = config['YMAX'] * config['hosfrac'] * config['ICufrac']
            y_obs = data[:, I_ICU]
        elif calmode == S_HOSCUM:
            title = 'Hospitalized Cumulative'
            ymax = config['YMAX'] * config['hosfrac'] * 2
            y_obs = data[:, I_HOSCUM]
        elif calmode == S_DEAD:
            title = 'Fatalities'
            ymax = config['YMAX'] * config['hosfrac'] * config['dfrac'] * 2
            y_obs = data[:, I_DEAD]
        else:
            ymax = config['YMAX']
        prior = results['fw'][0]
        transparancy = min(5.0 / len(prior), 1)
        print(calmode, output_index)
        prior_curves = np.array([member[output_index, :] for member in prior]).T
        time = fwd_args['time'] - fwd_args['time_delay']
        prior_mean = np.mean(prior_curves, axis=-1)
        color = 'green'
        plt.scatter(t_obs, y_obs, marker='o', c='k', label='Data')
        plt.plot(time, prior_curves, alpha=transparancy, c=color)
        plt.plot(time, prior_mean, lw=2, c=color, label='Mean of prior')
        plt.xlim(0, config['XMAX'])
        plt.ylim(0, ymax)
        plt.xlabel('Time [days]')
        plt.ylabel('Number of cases')
        plt.legend(loc='upper left')
        plt.title(title + ' prior ensemble')
        plt.savefig('{}_prior_ensemble_{}.png'.format(outpath, calmode), dpi=300)
        plt.close()

        # Plot posterior
        posterior = results['fw'][-1]
        posterior_curves = np.array([member[output_index, :] for member in posterior]).T
        time = fwd_args['time'] - fwd_args['time_delay']
        post_mean = np.mean(posterior_curves, axis=-1)
        post_med = np.median(posterior_curves, axis=-1)
        color = 'blue'
        plt.scatter(t_obs, y_obs, marker='o', c='k', label='Data')
        plt.plot(time, posterior_curves, alpha=transparancy, c=color)
        plt.plot(time, post_mean, lw=2, ls=':', c=color, label='Mean of posterior')
        plt.plot(time, post_med, lw=2, c=color, label='Median of posterior')
        plt.xlim(0, config['XMAX'])
        plt.ylim(0, ymax)
        plt.xlabel('Time [days]')
        plt.ylabel('Number of cases')
        plt.legend(loc='upper left')
        plt.title(title + ' posterior ensemble')
        plt.savefig('{}_posterior_ensemble_{}.png'.format(outpath, calmode), dpi=300)

        plt.xlim(0, time.max())
        plt.savefig('{}_posterior_ensemble_{}_longterm.png'.format(outpath, calmode), dpi=300)
        plt.ylim(0, posterior_curves.max())
        plt.savefig('{}_posterior_ensemble_{}_longterm_alt.png'.format(outpath, calmode), dpi=300)
        plt.close()

        p_values = config['p_values']
        p_array = []
        steps = np.arange(1, time.max() + 1)
        t_ind = [np.where(time == a)[0][0] for a in steps]
        posterior_length = posterior_curves.shape[1]
        for post_day in posterior_curves[t_ind, :]:
            array_sorted = np.sort(post_day)
            p_array.append([array_sorted[int(posterior_length * p)] for p in p_values])

        h_pvalues = ['P' + str(int(100 * a)) for a in p_values]
        header = 'time,mean,' + ','.join(h_pvalues) + ',observed'
        p_array = np.asarray(p_array)
        observed = np.pad(y_obs, (0, len(steps) - len(y_obs)), mode='constant', constant_values=np.nan)[:, None]
        table = np.concatenate((steps[:, None], post_mean[t_ind, None], p_array, observed), axis=1)
        np.savetxt('{}_posterior_prob_{}_calibrated_on_{}.csv'.format(outpath, calmode, config['calibration_mode']),
                   table, header=header, delimiter=',', comments='', fmt='%.2f')


        # save calibrated alfa values over time
        mnames = results['mnames']
        mvalues = results['mvalues']

        dayalpha1 = config['dayalpha']
        dayalpha = np.array(dayalpha1)
        alpha = np.zeros(np.size(dayalpha))
        alpha_sd = np.zeros(np.size(dayalpha))
        icount = 0
        for i, val in enumerate(mnames):
            if val == 'a':
                alpha[icount] = mvalues[i][2]
                alpha_sd[icount] = mvalues[i][3]
                icount += 1


        alpha_t = np.zeros_like(steps)
        alpha_sd_t = np.zeros_like(steps)
        for j, dayr in enumerate(dayalpha):
            alpha_t[steps.tolist().index(dayr):] = alpha[j]
            alpha_sd_t[steps.tolist().index(dayr):] = alpha_sd[j]

        p_values = [0.05,  0.5,   0.95]
        h_pvalues = ['P' + str(int(100 * a)) for a in p_values]
        header = 'time,' + ','.join(h_pvalues)
        p_array = np.asarray(p_array)
        observed = np.pad(y_obs, (0, len(steps) - len(y_obs)), mode='constant', constant_values=np.nan)[:, None]
        p5 = alpha_t + alpha_sd_t * 1.96
        p95 = alpha_t - alpha_sd_t * 1.96
        table = np.concatenate((steps[:,None],  p5[:,None], alpha_t[:, None],  p95[:,None]), axis=1)
        np.savetxt('{}_posterior_prob_{}_calibrated_on_{}.csv'.format(outpath, 'alpha', config['calibration_mode']),
                   table, header=header, delimiter=',', comments='', fmt='%.2f')


def main(configpath):
    # Load the model configuration file and the data (observed cases)
    config = load_config(configpath)
    data,firstdate = load_data(config)
    # concatenate additional column for actual observed hospitalized based
    t_obs = data[:, I_TIME]

    useworldfile = config['worldfile']
    if (not useworldfile):
        data = generate_hos_actual(data, config)
    else:
        data = generate_zero_columns(data, config)
        # Run the forward model to obtain a prior ensemble of models
    save_input_data(configpath, data)


    calibration_mode = config['calibration_mode']
    if calibration_mode == S_DEAD:
        y_obs = data[:, I_DEAD]
        output_index = O_DEAD
    elif calibration_mode == S_HOS:
        y_obs = data[:, I_HOS]
        output_index = O_HOS
    elif calibration_mode == S_ICU:
        y_obs = data[:, I_ICU]
        output_index = O_ICU
    elif calibration_mode == S_HOSCUM:
        y_obs = data[:, I_HOSCUM]
        output_index = O_HOSCUM
    else:
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
    m_prior, fwd_args = parse_config(config)
    m_prior = reshape_prior(m_prior)

    # Estimated uncertainty on the data
    # sigma = np.sqrt(y_obs).clip(min=10)
    error = config['observation_error']
    if calibration_mode == S_HOS or calibration_mode == S_HOSCUM:
        error = 2 * error
    sigma = error * np.ones_like(y_obs)

    add_arg = [fwd_args]
    kwargs = {'t_obs': t_obs,
              'output_index': output_index,
              'G': map_model_to_obs,
              'print_results': 1}

    results = im.es_mda(forward, m_prior, y_obs, sigma, *add_arg, **kwargs)
    plot_prior_and_posterior(results, fwd_args, config, configpath, t_obs, data, calibration_mode, output_index)
    base = (os.path.split(configpath)[-1]).split('.')[0]
    outpath = os.path.join(os.path.split(os.getcwd())[0], 'output', base + '_output.h5')
    save_results(results, fwd_args, config, outpath, data, mode='esmda')


if __name__ == '__main__':
    # This script only accepts one input argument: the path to the configuration .json file
    main(sys.argv[1])
