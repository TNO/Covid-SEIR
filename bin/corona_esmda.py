from src.coronaSEIR import base_seir_model
from src.esmda import InversionMethods
from src.io_func import load_data, load_config, hospital_forecast_to_txt, save_results, save_input_data, \
    add_hammer_to_results
from src.parse import parse_config, reshape_prior, get_mean
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from src.tools import generate_hos_actual, generate_zero_columns, calc_axis_interval
import datetime
import matplotlib.dates as mdates
import copy
from tqdm import tqdm

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
O_CUMINF = 11
O_HOSCUMICU = -1

S_HOS = 'hospitalized'
S_ICU = 'ICU'
S_HOSCUM = 'hospitalizedcum'
S_DEAD = 'dead'
S_INF = 'infected'

s_calmodes = [S_DEAD, S_HOS, S_ICU, S_HOSCUM, S_INF]
i_calmodes = [I_DEAD, I_HOS, I_ICU, I_HOSCUM, I_INF]
o_calmodes = [O_DEAD, O_HOS, O_ICU, O_HOSCUM, O_CUMINF]


def get_calibration_modes(dict):
    i_calibration_modes = np.array(dict)
    try:
        # try if it is array input
        a = i_calibration_modes[0]
    except:
        # if not return as array with single element
        print(' input calibration_mode not as array:  reformatting [ "', dict, '"]')
        return [dict]

    calibration_modes = i_calibration_modes
    if (i_calibration_modes.dtype == 'int32'):
        n = np.size(i_calibration_modes)
        calibration_modes = np.ones(n, dtype=np.object)
        for i, icolumn in enumerate(i_calibration_modes):
            for j, ical in enumerate(i_calmodes):
                if (ical == icolumn):
                    calibration_modes[i] = s_calmodes[j]

    return calibration_modes


def get_calibration_errors(dict):
    calibration_errors = np.array(dict)
    try:
        # try if it is array input
        a = calibration_errors[0]
    except:
        # if not return as array with single element
        print(' input observation errors not as array:  reformatting [ "', dict, '"]')
        return [dict]
    return calibration_errors


def map_model_to_obs(fwd, t_obs, output_index):
    """"
    #    return the modelled values for all t_obs
    #     # Added: Recovered, Hspitalized, Dead
    :param fwd: the forward model results
    :param t_obs: the observation times to collect the results
    :param output_index: array defining the output indices to be used
    :return:
    """

    # old code
    # t_ind = [np.where(fwd[O_TIME, :] == a)[0][0] for a in t_obs]
    # return fwd[output_index, t_ind]

    t_ind = [np.where(fwd[O_TIME, :] == a)[0][0] for a in t_obs]

    retval = fwd[output_index[0], t_ind]
    for i, num in enumerate(output_index):
        if (i > 0):
            r = fwd[num, t_ind]
            retval = np.append(retval, r)
    return retval


def plot_prior_and_posterior(results, fwd_args, config, configpath, t_obs, data, calibration_mode, output_index):
    # Plot prior
    base = (os.path.split(configpath)[-1]).split('.')[0]
    outpath = os.path.join(os.path.split(os.getcwd())[0], 'output', base)

    date_1 = datetime.datetime.strptime(config['startdate'], "%m/%d/%y")
    t_obs = [date_1 + datetime.timedelta(days=a - 1) for a in t_obs]

    calmodes = [S_HOS, S_ICU, S_HOSCUM, S_DEAD, S_INF]
    o_indices = [O_HOS, O_ICU, O_HOSCUM, O_DEAD, O_CUMINF]
    # useworldfile = config['worldfile']
    # if (useworldfile):
    #    calmodes = [ S_DEAD]
    #    o_indices = [ O_DEAD]
    # else:
    #    # check if I_HOS is present in the data, if not generate it
    #    print('number of columns in data ', data.shape[1])
    #    print('number of rows in data ', data.shape[0])
    #    ncol = data.shape[1]

    #    data = generate_hos_actual(data, config)

    titles = ['Hospitalized', 'ICU', 'Hospitalized cum', 'Mortalities', 'infected']
    # ['lightcoral', 'brown']
    # ['mistyrose', 'lightcoral'],
    symcolors = [['powderblue', 'steelblue'], ['peachpuff', 'sandybrown'], ['lightgreen', 'forestgreen'],
                 ['gainsboro', 'silver'], ['mistyrose', 'lightcoral']]
    y_obs_s = [data[:, I_HOS], data[:, I_ICU], data[:, I_HOSCUM], data[:, I_DEAD], data[:, I_INF]]
    y_maxdef = config['YMAX']
    y_maxhos = y_maxdef * get_mean(config['hosfrac'])
    y_maxicu = y_maxhos * get_mean(config['ICufrac'])
    y_maxdead = y_maxhos * get_mean(config['dfrac']) * 4
    y_maxinf = y_maxdef * 10
    y_max = [y_maxhos, y_maxicu, y_maxhos * 4, y_maxdead, y_maxinf]

    casename = ''
    try:
        casename = config['plot']['casename']
    except:
        print('No casename in plot parameters')
        pass

    for i, calmode in enumerate(calmodes):
        output_index = o_indices[i]
        title = titles[i]
        symcolor = symcolors[i]
        y_obs = y_obs_s[i]
        ymax = y_max[i]

        prior = results['fw'][0]
        transparancy = min(5.0 / len(prior), 1)
        print(calmode, output_index)
        prior_curves = np.array([member[output_index, :] for member in prior]).T
        time = fwd_args['time'] - fwd_args['time_delay']
        times = [date_1 + datetime.timedelta(days=a - 1) for a in time]
        prior_mean = np.mean(prior_curves, axis=-1)
        color = 'green'
        plt.scatter(t_obs, y_obs, marker='o', c='k', label='Data')
        plt.plot(times, prior_curves, alpha=transparancy, c=color)
        plt.plot(times, prior_mean, lw=2, c=color, label='Mean of prior')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        day_interval = calc_axis_interval((times[config['XMAX']] - date_1).days)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
        plt.xlim(date_1, times[config['XMAX']])
        plt.ylim(0, ymax)
        plt.xlabel('Time [days]')
        plt.gcf().autofmt_xdate()
        plt.ylabel('Number of cases')
        plt.legend(loc='upper left')
        plt.title(title + ' prior ensemble')
        plt.savefig('{}_prior_ensemble_{}.png'.format(outpath, calmode), dpi=300)
        plt.close()

        # Plot posterior
        posterior = results['fw'][-1]
        posterior_curves = np.array([member[output_index, :] for member in posterior]).T
        post_mean = np.mean(posterior_curves, axis=-1)
        post_med = np.median(posterior_curves, axis=-1)
        color = 'blue'
        plt.scatter(t_obs, y_obs, marker='o', c='k', label='Data')
        plt.plot(times, posterior_curves, alpha=transparancy, c=color)
        plt.plot(times, post_mean, lw=2, ls=':', c=color, label='Mean of posterior')
        plt.plot(times, post_med, lw=2, c=color, label='Median of posterior')
        plt.xlim(date_1, times[config['XMAX']])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        day_interval = calc_axis_interval((times[config['XMAX']] - date_1).days)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
        plt.gcf().autofmt_xdate()
        plt.ylim(0, ymax)
        plt.xlabel('Time [days]')
        plt.ylabel('Number of cases')
        plt.legend(loc='upper left')
        plt.title(title + ' posterior ensemble')
        plt.savefig('{}_posterior_ensemble_{}.png'.format(outpath, calmode), dpi=300)

        plt.xlim(date_1, times[-1])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        day_interval = calc_axis_interval((times[-1] - date_1).days)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
        plt.gcf().autofmt_xdate()
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
    # posterior
    alpha = np.zeros(np.size(dayalpha))
    alpha_sd = np.zeros(np.size(dayalpha))
    # priors
    alpha0 = np.zeros(np.size(dayalpha))
    alpha0_sd = np.zeros(np.size(dayalpha))
    icount = 0
    for j, val in enumerate(mnames):
        if val == 'a':
            alpha0[icount] = mvalues[j][0]
            alpha0_sd[icount] = mvalues[j][1]
            alpha[icount] = mvalues[j][2]
            alpha_sd[icount] = mvalues[j][3]
            icount += 1

    names = ['posterior', 'prior']
    for j in range(0, 2):
        alpha_t = np.zeros_like(steps)
        alpha_sd_t = np.zeros_like(steps)
        for k, dayr in enumerate(dayalpha):
            if (j == 0):
                alpha_t[steps.tolist().index(dayr):] = alpha[k]
                alpha_sd_t[steps.tolist().index(dayr):] = alpha_sd[k]
            else:
                alpha_t[steps.tolist().index(dayr):] = alpha0[k]
                alpha_sd_t[steps.tolist().index(dayr):] = alpha0_sd[k]

        p_values = [0.05, 0.25, 0.5, 0.75, 0.95]
        h_pvalues = ['P' + str(int(100 * a)) for a in p_values]
        header = 'time,' + ','.join(h_pvalues)
        p_array = np.asarray(p_array)
        observed = np.pad(y_obs, (0, len(steps) - len(y_obs)), mode='constant', constant_values=np.nan)[:, None]
        p5 = alpha_t + alpha_sd_t * 1.96
        p95 = alpha_t - alpha_sd_t * 1.96
        p25 = alpha_t + alpha_sd_t * 0.69
        p75 = alpha_t - alpha_sd_t * 0.69
        table = np.concatenate((steps[:, None], p5[:, None], p25[:, None], alpha_t[:, None], p75[:, None],
                                p95[:, None]), axis=1)
        np.savetxt('{}_{}_prob_{}_calibrated_on_{}.csv'.format(
            outpath, names[j], 'alpha', config['calibration_mode']),
            table, header=header, delimiter=',', comments='', fmt='%.2f')


def apply_hammer(posterior_fw, posterior_param, fwd_args, hammer_icu, hammer_alpha, data_end):
    # For each ensemble member, see when the hammer_ICU value is crossed (upward trend)
    # Stop
    hammer_time = []
    base_day_alpha = fwd_args['locked']['dayalpha']
    for p_ind, member in enumerate(posterior_fw):
        time = member[O_TIME, :]
        time_mask = time > data_end
        time = member[O_TIME, time_mask]
        icu = member[O_ICU, time_mask]
        icu_slope = np.concatenate((np.diff(icu), [-1]))
        hammer = np.logical_and(icu >= hammer_icu, icu_slope > 0)
        try:
            hammer_time.append(min(time[hammer]))
        except ValueError:  # If hammer condition is never met in this realization:
            hammer_time.append(-1)

    hammer_time = np.array(hammer_time)
    alpha_offset = (fwd_args['free_param']).index('alpha')
    new_param = []
    new_fwd_args = []
    for h_ind, ht in enumerate(hammer_time):
        if ht > 0:
            day_alpha = copy.deepcopy(base_day_alpha)
            day_alpha[day_alpha >= ht] = ht
            param = copy.deepcopy(posterior_param[h_ind])
            new_alpha = np.random.uniform(*hammer_alpha)
            change_alpha = np.where(day_alpha == ht)[0] + alpha_offset
            param[change_alpha] = new_alpha
            n_fwd = copy.deepcopy(fwd_args)
            n_fwd['locked']['dayalpha'] = day_alpha
            new_param.append(param)
            new_fwd_args.append(n_fwd)
        else:
            new_param.append(posterior_param[h_ind])
            new_fwd_args.append(copy.deepcopy(fwd_args))

    # Run the updated posterior (with hammer applied) again
    result = []
    for p_ind, param in enumerate(tqdm(new_param, desc='Running hammered posterior')):
        fwd = new_fwd_args[p_ind]
        result.append(base_seir_model(param, fwd))

    return result


def main(configpath):
    # Load the model configuration file and the data (observed cases)
    config = load_config(configpath)
    data = load_data(config)
    # concatenate additional column for actual observed hospitalized based
    t_obs = data[:, I_TIME]

    useworldfile = config['worldfile']
    if (not useworldfile):
        data = generate_hos_actual(data, config)
    else:
        data = generate_zero_columns(data, config)
        # Run the forward model to obtain a prior ensemble of models
    save_input_data(configpath, data)

    calibration_modes = get_calibration_modes(config['calibration_mode'])
    observation_errors = get_calibration_errors(config['observation_error'])
    calibration_mode = calibration_modes[0]

    found = False
    for i, calmode in enumerate(s_calmodes):
        if (calmode == calibration_mode):
            print('ensemble fit on : ', calibration_modes[0], ' with error ', observation_errors[0])
            y_obs = data[:, i_calmodes[i]]
            output_index = [o_calmodes[i]]
            error = np.ones_like(y_obs) * observation_errors[0]
            found = True
    if not found:
        raise NotImplementedError

    for ical, calibration_mode in enumerate(calibration_modes):
        if (ical > 0):
            print('additional ensemble fit on : ', calibration_modes[ical], ' with error ', observation_errors[ical])
            for i, calmode in enumerate(s_calmodes):
                if (calmode == calibration_mode):
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
    m_prior, fwd_args = parse_config(config)
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

    plot_prior_and_posterior(results, fwd_args, config, configpath, t_obs, data, calibration_mode, output_index)
    base = (os.path.split(configpath)[-1]).split('.')[0]
    outpath = os.path.join(os.path.split(os.getcwd())[0], 'output', base + '_output.h5')
    save_results(results, fwd_args, config, outpath, data, mode='esmda')

    # Check if we want to apply a 'hammer'
    try:
        hammer_icu = config['hammer_ICU']
        hammer_alpha = config['hammer_alpha']
        assert len(hammer_alpha) == 2
        hammered_results = apply_hammer(results['fw'][-1], results['M'][-1], fwd_args, hammer_icu, hammer_alpha,
                                        data_end=data[-1, 0])
        # TODO: Add hammer results to h5 file and plotting
        add_hammer_to_results(hammered_results, outpath, mode='esmda')

    except KeyError as e:
        pass  # Don't apply hammer, you're done early!


if __name__ == '__main__':
    # This script only accepts one input argument: the path to the configuration .json file
    main(sys.argv[1])
