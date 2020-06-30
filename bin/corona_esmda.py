from src import coronaSEIR
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
O_ALPHARUN = 12
O_HOSCUMICU = -1

S_HOS = 'hospitalized'
S_ICU = 'ICU'
S_HOSCUM = 'hospitalizedcum'
S_DEAD = 'dead'
S_INF = 'infected'
S_ALPHARUN = 'alpharun'

s_calmodes = [S_DEAD, S_HOS, S_ICU, S_HOSCUM, S_INF]
i_calmodes = [I_DEAD, I_HOS, I_ICU, I_HOSCUM, I_INF]
o_calmodes = [O_DEAD, O_HOS, O_ICU, O_HOSCUM, O_CUMINF]


def single_run_mean(config, data):
    ndata = np.size(data[:, 0])
    m_prior, fwd_args = parse_config(config, ndata, mode='mean')
    m_prior = reshape_prior(m_prior)
    param = m_prior[0]
    res = base_seir_model(param, fwd_args)

    try:
        posterior_param = [param]
        accc_timeinterval = config['ACCC_timeinterval']
        accc_timestart = config['ACCC_timestart']
        accc_step = config['ACCC_step']
        accc_maxstep = config['ACCC_maxstep']
        accc_step_sd = config['ACCC_step_sd']
        accc_low = config['ACCC_low']
        accc_slope = config['ACCC_slope']
        accc_scale = config['ACCC_scale']
        accc_cliplow = config['ACCC_cliplow']
        accc_cliphigh = config['ACCC_cliphigh']
        time_delay = config['time_delay']
        hammer_icu = config['hammer_ICU']
        hammer_slope = config['hammer_slope']
        hammer_release = config['hammer_release']
        hammer_alpha = config['hammer_alpha']
        accc_results = apply_accc(config['output_base_filename'], posterior_param, fwd_args, accc_timeinterval,
                                  accc_timestart, accc_maxstep, accc_step, accc_step_sd, accc_low, accc_slope,
                                  accc_scale, hammer_icu, hammer_slope, hammer_alpha, hammer_release, accc_cliplow,
                                  accc_cliphigh, time_delay, data_end=data[-1, 0])
        res = accc_results[0]
    except KeyError:
        pass

    return res


def find_N(config, data, imatch=I_DEAD, itarget=10):
    """
    Create a prior ensemble of model runs.
    :param config: The entire model configuration in dict-form
    :param  imatch:  column in data to match N
    :param  itarget: number of target value to match N
    :return: N to use
    """

    np.random.seed(1)
    # Parse parameters
    #  get just 1 base case sample corresponding to average

    # results = np.zeros((len(m_prior), 13, len(fwd_args['time'])))

    i_mod = o_calmodes[imatch]
    i_obs = i_calmodes[imatch]
    obsdead = data[:, i_obs]
    time_delay = config['time_delay']
    obsdead_index = np.where(obsdead > itarget)[0][0] + time_delay
    found = False
    icount = 0
    ncountmax = 50
    nnew = 1000

    ndata = np.size(data[:, 0])
    m_prior, fwd_args = parse_config(config, ndata, mode='mean')
    m_prior = reshape_prior(m_prior)
    param = m_prior[0]

    while not found and icount < ncountmax:
        fwd_args['locked']['n'] = nnew
        res = base_seir_model(param, fwd_args)
        moddead = res[i_mod, :]
        moddead_index = np.where(moddead > itarget)

        print('moddead index, obsdead index ', moddead_index[0][0], obsdead_index)
        found = moddead_index[0][0] >= obsdead_index
        if not found:
            icount += 1
            nnew = fwd_args['locked']['n'] * 2
            fwd_args['locked']['n'] = nnew

    return nnew


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
    if i_calibration_modes.dtype == 'int32':
        n = np.size(i_calibration_modes)
        calibration_modes = np.ones(n, dtype=np.object)
        for i, icolumn in enumerate(i_calibration_modes):
            for j, ical in enumerate(i_calmodes):
                if ical == icolumn:
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
        if i > 0:
            r = fwd[num, t_ind]
            retval = np.append(retval, r)
    return retval


def save_and_plot_prior_and_posterior(results, fwd_args, config, base_filename, t_obs, data, calibration_mode,
                                      output_index, save_plots, save_files, plothammer=False):
    # Save plots, CSV's, and return the organized data

    # Initialize variables
    outpath = os.path.join(os.path.split(os.getcwd())[0], 'output', base_filename)
    calmodes = [S_HOS, S_ICU, S_HOSCUM, S_DEAD, S_INF, S_ALPHARUN]
    o_indices = [O_HOS, O_ICU, O_HOSCUM, O_DEAD, O_CUMINF, O_ALPHARUN]
    y_obs_s = [data[:, I_HOS], data[:, I_ICU], data[:, I_HOSCUM], data[:, I_DEAD], data[:, I_INF],[]]
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

    prior = results['fw'][0]
    posterior = results['fw'][-1]
    date_1 = datetime.datetime.strptime(config['startdate'], "%m/%d/%y")
    time = fwd_args['time'] - fwd_args['time_delay']
    times = [date_1 + datetime.timedelta(days=a - 1) for a in time]
    p_values = config['p_values']
    steps = np.arange(1, time.max() + 1)
    t_ind = [np.where(time == a)[0][0] for a in steps]
    h_pvalues = ['P' + str(int(100 * a)) for a in p_values]
    header = 'time,mean,' + ','.join(h_pvalues) + ',observed'

    # Initialize variables for the plots
    if save_plots:
        t_obs = [date_1 + datetime.timedelta(days=a - 1) for a in t_obs]

        titles = ['Hospitalized', 'ICU', 'Hospitalized Cum.', 'Mortalities', 'Infected', '1-$\\alpha$']
        y_label = ['Number of cases', 'Number of cases', 'Number of cases',
                   'Number of cases', 'Number of cases', '1-$\\alpha$']
        symcolors = [['powderblue', 'steelblue'], ['peachpuff', 'sandybrown'], ['lightgreen', 'forestgreen'],
                     ['silver', 'grey'], ['mistyrose', 'lightcoral'], ['violet', 'purple']]
        transparency = max(min(5.0 / len(prior), 1), 0.05)

        y_maxdef = config['YMAX']
        y_maxhos = y_maxdef * get_mean(config['hosfrac'])
        y_maxicu = y_maxhos * get_mean(config['ICufrac'])
        y_maxdead = y_maxhos * get_mean(config['dfrac']) * 4
        y_maxinf = y_maxdef * 10
        y_max = [y_maxhos, y_maxicu, y_maxhos * 4, y_maxdead, y_maxinf, 1.0]

        casename = ''
        try:
            casename = config['plot']['casename']
        except:
            print('No casename in plot parameters')
            pass

    posterior_prior_data = {}
    for i, calmode in enumerate(calmodes):
        output_index = o_indices[i]
        y_obs = y_obs_s[i]
        print(calmode, output_index)

        # Prepare prior data
        prior_curves = np.array([member[output_index, :] for member in prior]).T
        prior_mean = np.mean(prior_curves, axis=-1)

        # Prepare posterior data
        posterior_curves = np.array([member[output_index, :] for member in posterior]).T
        post_mean = np.mean(posterior_curves, axis=-1)
        post_med = np.median(posterior_curves, axis=-1)

        if save_plots:
            title = titles[i]
            symcolor = symcolors[i]
            ymax = y_max[i]
            ylabel = y_label[i]

            # Save prior plot
            color = 'green'
            if np.size(y_obs) > 0:
                plt.scatter(t_obs, y_obs, marker='o', c='k', label='Data')
            plt.plot(times, prior_curves, alpha=transparency, c=color)
            plt.plot(times, prior_mean, lw=2, c=color, label='Mean of prior')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            day_interval = calc_axis_interval((times[config['XMAX']] - date_1).days)
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
            plt.xlim(date_1, times[config['XMAX']])
            plt.ylim(0, ymax)
            plt.xlabel('Date')
            plt.gcf().autofmt_xdate()
            plt.ylabel(ylabel)
            plt.legend(loc='upper left')
            plt.title(title + ' prior ensemble')
            plt.savefig('{}_prior_ensemble_{}.png'.format(outpath, calmode), dpi=300)
            plt.close()

            # Plot posterior
            isicu = (calmode == S_ICU)
            try:
                figure_size = config['plot']['figure_size']
                assert len(figure_size) == 2
                plt.figure(figsize=figure_size)
            except:
                pass
            if (isicu) and plothammer:
                try:
                # try to read hammer
                # consider to plot the analyse_hammer results
                    path = '{}_hammer_diagnostics{}.csv'.format(outpath, '' '')
                    data = np.genfromtxt(path, names=True, delimiter=',')
                    th_trig = data['daymon']
                    h_icurate = data['icurate']
                    h_icu = data['icu']
                    th_max = data['daymax']
                    h_icupeak = data['icumax']
                    slope_max = config['hammer_slope']
                    th_plot = [date_1 + datetime.timedelta(days=a - 1) for a in th_trig]
                    plt.scatter(th_plot, h_icu,  c=h_icurate, vmin=0, vmax=slope_max, marker='o', s=8)
                    th_plot = [date_1 + datetime.timedelta(days=a - 1) for a in th_max]
                    plt.scatter(th_plot, h_icupeak, c=h_icurate, vmin=0, vmax=slope_max, marker='o', s=8)
                    cbar = plt.colorbar()
                    cbar.set_label("daily ICU", labelpad=+1)
                except:
                    print('no hammer entries in ', path)
                    pass

            # Save posterior plots
            color = 'blue'
            if np.size(y_obs) > 0:
                plt.scatter(t_obs, y_obs, marker='o', c='k', label='Data')
            plt.plot(times, posterior_curves, alpha=transparency, c=color)
            plt.plot(times, post_mean, lw=2, ls=':', c=color, label='Mean of posterior')
            plt.plot(times, post_med, lw=2, c=color, label='Median of posterior')
            plt.xlim(date_1, times[config['XMAX']])
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            day_interval = calc_axis_interval((times[config['XMAX']] - date_1).days)
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
            plt.gcf().autofmt_xdate()
            plt.ylim(0, ymax)
            plt.xlabel('Date')
            plt.ylabel(ylabel)
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

        # Save results to CSV
        posterior_prior_data[calmode] = save_posterior_and_prior(posterior_curves, t_ind, p_values, y_obs, steps,
                                                                 post_mean, outpath, calmode, config, header,
                                                                 save_files)

    # Save alpha results to CSV
    posterior_prior_data['alpha'] = save_prior_and_posterior_alpha(results, config, steps, outpath, save_files)

    return posterior_prior_data


def save_posterior_and_prior(posterior_curves, t_ind, p_values, y_obs, steps, post_mean,
                             outpath, calmode, config, header, save_files):
    p_array = []
    posterior_length = posterior_curves.shape[1]
    for post_day in posterior_curves[t_ind, :]:
        array_sorted = np.sort(post_day)
        p_array.append([array_sorted[int(posterior_length * p)] for p in p_values])

    p_array = np.asarray(p_array)
    observed = np.pad(y_obs, (0, len(steps) - len(y_obs)), mode='constant', constant_values=np.nan)[:, None]
    table = np.concatenate((steps[:, None], post_mean[t_ind, None], p_array, observed), axis=1)
    if save_files:
        np.savetxt('{}_posterior_prob_{}_calibrated_on_{}.csv'.format(outpath, calmode, config['calibration_mode']),
                   table, header=header, delimiter=',', comments='', fmt='%.2f')
    return table


def save_prior_and_posterior_alpha(results, config, steps, outpath, save_files):
    # save prior and posterior alpha values over time, according to saved parameters (this is not any more necessary
    # when we have these from the run)

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
        if val == 'alpha':
            alpha0[icount] = mvalues[j][0]
            alpha0_sd[icount] = mvalues[j][1]
            alpha[icount] = mvalues[j][2]
            alpha_sd[icount] = mvalues[j][3]
            icount += 1

    names = ['posterior', 'prior']
    table = {}
    for j in range(0, 2):
        alpha_t = np.zeros_like(steps)
        alpha_sd_t = np.zeros_like(steps)
        for k, dayr in enumerate(dayalpha):
            if j == 0:
                alpha_t[steps.tolist().index(dayr):] = alpha[k]
                alpha_sd_t[steps.tolist().index(dayr):] = alpha_sd[k]
            else:
                alpha_t[steps.tolist().index(dayr):] = alpha0[k]
                alpha_sd_t[steps.tolist().index(dayr):] = alpha0_sd[k]

        p_values = [0.05, 0.25, 0.5, 0.75, 0.95]
        h_pvalues = ['P' + str(int(100 * a)) for a in p_values]
        header = 'time,' + ','.join(h_pvalues)
        p5 = alpha_t + alpha_sd_t * 1.96
        p95 = alpha_t - alpha_sd_t * 1.96
        p25 = alpha_t + alpha_sd_t * 0.69
        p75 = alpha_t - alpha_sd_t * 0.69
        table[names[j]] = np.concatenate((steps[:, None], p5[:, None], p25[:, None], alpha_t[:, None], p75[:, None],
                                          p95[:, None]), axis=1)
        if save_files:
            np.savetxt('{}_{}_prob_{}_calibrated_on_{}.csv'.format(
                outpath, names[j], 'alpha', config['calibration_mode']),
                table[names[j]], header=header, delimiter=',', comments='', fmt='%.2f')
    return table


def signal_hammer(icu, icu_slope, hammer_icu, hammer_slope):
    """"
    #    check if at the current icu and icu_slope (daily icu)
    #     a safety hammer should be applied
    :param icu: the current icu
    :param icu_slope: the current daily icu
    :param  hammer_icu: the icu level to activate the hammer
    :param  hammer_slope: the icu slope level to activate the hammer
    :return: boolean
    """

    return (icu > hammer_icu and icu_slope > 0) or (icu_slope > hammer_slope)


def get_alpha_scale(icu, icu_slope, icu_slope2, accc_low,  accc_scale=1, accc_slope=20):
    acc = 0.05 * accc_slope
    keep = 0.2 * accc_slope

    icu_dif = accc_low - icu
    # determine number of days to overcome dif, assuming slope is pointing the right way
    if icu_slope == 0:
        icu_slope = 1
    days = icu_dif / icu_slope
    if abs(icu_slope2)<1e-20:
        icu_slope2= 1e-20
    step = 0
    if days > 0:
        # slope in right directions, based on days to reach accc_low
        if icu_slope / icu_slope2 > 0:

            # self strengtherning, slow down  actively but allow low slope still to grow
            if abs(icu_slope) < acc:
                if abs(icu_slope2) < 0.1 * acc:
                   step = -np.sign(icu_slope)
                else:
                    step = 0
            elif abs(icu_slope) < keep:
               if abs(icu_slope2) < 0.1 * acc:
                   step = 0
                   # keep growing slope
               else:
                    step = np.sign(icu_slope)
            else:
                # slow down growing slope
                step = np.sign(icu_slope)
        else:
            # allow to grow, if the second derivative is noticable
            if abs(icu_slope2) > 0.1 * acc:
                step = -np.sign(icu_slope)
            else:
                step = 0
    else:
        # growth needed as slope is going the wrong way
        step = np.sign(icu_slope) * accc_scale
        # if it is also accelerating the wrong way put more effort in accelerating the right way (enlarging the step)
        if icu_slope / icu_slope2 > 0:
            step = step * accc_scale

    return step


def apply_accc(base_filename, posterior_param, fwd_args, accc_timeinterval, accc_timestart, accc_maxstep,
               accc_step, accc_step_sd, accc_low, accc_slope, accc_scale, hammer_icu, hammer_slope, hammer_alpha,
               hammer_release,accc_cliplow, accc_cliphigh,  time_delay,data_end):
    # For each ensemble member, see when the hammer_ICU value is crossed (upward trend)
    # Stop
    hammer_time = []
    base_day_alpha = fwd_args['locked']['dayalpha']
    # print('isample,hammertime,icu@hammer,icuslope@hammer,icumaxtime,icumax')

    #day_mon = int (data_end +time_delay)

    #day_mon += (accc_timestart - data_end)

    alpha_offset = (fwd_args['free_param']).index('alpha')

    result = []

    table_diagnostics = []
    table_hammer_diagnostics = []
    # numberofdays forward in baseSEIR
    ndaysforward = 800
    ndaysforward = max(accc_timeinterval,ndaysforward)

    # for p_ind, member in enumerate(posterior_param):
    for p_ind in tqdm(range(len(posterior_param)), desc='running ACCC posterior'):
        day_alpha = copy.deepcopy(base_day_alpha)

        day_mon_start = int(data_end + time_delay)
        day_mon_start += int(accc_timestart - data_end)
        day_mon_start_accc = max(day_mon_start, day_alpha[-1])  #2*accc_timeinterval

        # run default forward just in case
        res = base_seir_model(posterior_param[p_ind], fwd_args)
        time = res[O_TIME, :]
        time_end = time[-1]
        # start with a copy of the base_day_alpha
        param = copy.deepcopy(posterior_param[p_ind])
        n_fwd = copy.deepcopy(fwd_args)
        alpha_scale_last = 0

        nlookahead = 7  # 7 #accc_timeinterval

        day_mon = day_mon_start

        #ndayshammer = 28
        #lastday_hammer = day_mon-ndayshammer
        # flag hammer has been released
        lastday_hammer = -1
        isteps = 0
        alpha_scale_sum = 0
        while day_mon < time_end:

            accc_ok = (day_mon - day_mon_start_accc) >= 0
            icu = res[O_ICU, :]
            icu_slope = np.concatenate((np.diff(icu), [-1]))
            icu_slope = np.roll(icu_slope, 1)
            icu_slope2 = icu_slope[day_mon + nlookahead] - icu_slope[day_mon + nlookahead-1]

            # check if the hammer needs to be applied
            if lastday_hammer > 0:
                if icu[day_mon]<hammer_release and icu_slope[day_mon] < 0:
                    lastday_hammer = -1
            donothing = (lastday_hammer > 0)
            #donothing = ((day_mon- lastday_hammer) < ndayshammer )
            dohammer = signal_hammer(icu[day_mon], icu_slope[day_mon], hammer_icu, hammer_slope)
            if donothing:
                pass
            elif dohammer:
                ht = day_mon
                a = hammer_alpha
                hammer_mean = 0.5 * (a[0] + a[1])
                hammer_sd = ((1.0 / np.sqrt(12)) * (a[1] - a[0]))
                hammer_alpha_normal = [hammer_mean, hammer_sd]
                alphastep = np.random.normal(hammer_mean,  hammer_sd)
                fwd_freeparam1 = n_fwd['free_param'][0:alpha_offset]

                fwd_alfa = n_fwd['free_param'][alpha_offset:len(day_alpha)+alpha_offset]
                fwd_freeparam2 = n_fwd['free_param'][len(day_alpha)+alpha_offset:]
                freeparamdict = copy.deepcopy(fwd_freeparam1)
                for i, element in enumerate(fwd_alfa):
                    freeparamdict.append(element)
                    # add one more alpha
                    if i == 0:
                        freeparamdict.append(element)
                for i, element in enumerate(fwd_freeparam2):
                    freeparamdict.append(element)

                # append day alpha
                day_alpha = np.append(day_alpha, ht)
                change_alpha = alpha_offset + len(day_alpha)-1
                alphanew = alphastep

                # restructure param to include alphanew
                param = np.concatenate((param[0:change_alpha], [alphanew], param[change_alpha:]), axis=None)

                n_fwd['locked']['dayalpha'] = day_alpha
                n_fwd['free_param'] = freeparamdict

                # res = base_seir_model(param, n_fwd, trestart= day_mon, tend= day_mon+ndaysforward, lastresult=res)
                res = base_seir_model(param, n_fwd, trestart=day_mon,  lastresult=res)

                alpha_scale_last = 0
                # collect hammer diagnostics
                icu = res[O_ICU, :]
                icuconsider = icu[day_mon:day_mon+ndaysforward]
                icumax = max(icuconsider)
                icumin = min(icuconsider)
                day_max = np.where(icuconsider >= icumax)[0][0] - time_delay + day_mon
                day_min = np.where(icuconsider <= icumin)[0][0] - time_delay + day_mon
                row_diagnostics = np.array([p_ind, day_mon-time_delay, day_min, day_max, icu[day_mon],
                                            icu_slope[day_mon], icumin, icumax, alphastep])
                table_hammer_diagnostics.append(row_diagnostics)
                lastday_hammer = day_mon
                alpha_scale_sum = 0
                #  check if it is time to adapt the ACCC
            elif accc_ok and ((day_mon-day_mon_start) % accc_timeinterval) == 0:

                model_forecast = False
                if model_forecast:
                    icu_forecast = icu[day_mon + nlookahead]
                    icu_slope_forecast = icu_slope[day_mon + nlookahead]
                    icu_slope2_forecast = icu_slope[day_mon + nlookahead] - icu_slope[day_mon + nlookahead - 1]
                else:
                    icu_slope2_forecast = icu_slope[day_mon] - icu_slope[day_mon - 1]
                    icu_slope_forecast = icu_slope[day_mon] + nlookahead*icu_slope2_forecast
                    # use second order taylor approximation for forecast
                    icu_forecast = icu[day_mon] + nlookahead * icu_slope[day_mon] + \
                                   0.5 * (nlookahead**2) * icu_slope2_forecast

                alpha_scale = get_alpha_scale (icu_forecast, icu_slope_forecast, icu_slope2_forecast,
                                               accc_low,  accc_scale, accc_slope)

                alpha_scale_sum += alpha_scale
                # if (alpha_scale_last==alpha_scale):
                #    alpha_scale = 0
                alphastep = 0
                if alpha_scale != 0 and alpha_scale_sum>= - accc_maxstep:
                    # change the last active alpha active before daymon

                    ht = day_mon
                    # sign = 0
                    alphastep = np.random.normal(accc_step * alpha_scale, accc_step_sd*abs(alpha_scale))
                    # print('pind step at day, level icu,slope, sign, alphastep ', p_ind, ht,
                    #       icu[day_mon], icu_slope[day_mon], sign, alphastep)

                    fwd_freeparam1 = n_fwd['free_param'][0:alpha_offset]
                    # last_alpha = np.where(day_alpha == ht)[0] + alpha_offset

                    fwd_alfa = n_fwd['free_param'][alpha_offset:len(day_alpha) + alpha_offset]
                    fwd_freeparam2 = n_fwd['free_param'][len(day_alpha) + alpha_offset:]
                    freeparamdict = copy.deepcopy(fwd_freeparam1)
                    for i, element in enumerate(fwd_alfa):
                        freeparamdict.append(element)
                        # add one more alpha
                        if i == 0:
                            freeparamdict.append(element)
                    for i, element in enumerate(fwd_freeparam2):
                        freeparamdict.append(element)

                    # append day alpha
                    day_alpha = np.append(day_alpha, ht)
                    change_alpha = alpha_offset + len(day_alpha)-1
                    alphanew = param[change_alpha - 1] + alphastep
                    alphanew = max(accc_cliplow, min(accc_cliphigh, alphanew))

                    # restructure param to include alphanew
                    param = np.concatenate((param[0:change_alpha], [alphanew], param[change_alpha:]), axis=None)

                    n_fwd['locked']['dayalpha'] = day_alpha
                    n_fwd['free_param'] = freeparamdict

                    #res = base_seir_model(param, n_fwd, trestart= day_mon, tend= day_mon+ndaysforward, lastresult=res)
                    res = base_seir_model(param, n_fwd, trestart=day_mon,  lastresult=res)
                    #res = base_seir_model(param, n_fwd)

                alpha_scale_last = alpha_scale

                icuconsider = icu[day_mon-accc_timeinterval:day_mon+1]
                icumax = max (icuconsider)
                icumin = min(icuconsider)
                day_max =  np.where(icuconsider>=icumax)[0][0] - time_delay - day_mon-accc_timeinterval
                day_min = np.where(icuconsider<= icumin)[0][0] - time_delay - day_mon-accc_timeinterval
                row_diagnostics = np.array ([p_ind, day_mon-time_delay, day_min, day_max,icu[day_mon], icu_slope[day_mon], icumin, icumax, alphastep])
                table_diagnostics.append(row_diagnostics)

            day_mon = day_mon + 1

        result.append(res)

    # save the diagnostics file of the
    outpath = os.path.join(os.path.split(os.getcwd())[0], 'output', base_filename)
    header = 'sampleid,daymon,daymin, daymax,icu,icurate,icumin,icumax,alphastep'
    table = np.array(table_diagnostics)
    # table =  table[:,0:5] #np.concatenate((datanew[:, 0:6]), axis=-1)
    np.savetxt('{}_accc_diagnostics{}.csv'.format(outpath, '', ''),
               table, header=header, delimiter=',', comments='', fmt='%.2f')
    table = np.array(table_hammer_diagnostics)
    # table =  table[:,0:5] #np.concatenate((datanew[:, 0:6]), axis=-1)
    np.savetxt('{}_hammer_diagnostics{}.csv'.format(outpath, '', ''),
               table, header=header, delimiter=',', comments='', fmt='%.2f')
    return result


def run_esmda_model(base_filename, config, data, save_plots=1, save_files=1):
    # concatenate additional column for actual observed hospitalized based
    t_obs = data[:, I_TIME]

    useworldfile = config['worldfile']
    if not useworldfile:
        data = generate_hos_actual(data, config)
    else:
        data = generate_zero_columns(data, config)
        # Run the forward model to obtain a prior ensemble of models
    save_input_data(base_filename, data)

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
                    error2[min(0,len(error2)-20):] *= 1
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

    # Run the find_N function only if specified in the config file
    try:
        if config['find_N']:
            nnew = find_N(config, data)
            config['N'] = {'type': 'uniform', 'min': 0.25 * nnew, 'max': 4 * nnew}
    except KeyError:
        pass

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

    prior_and_posterior_results = save_and_plot_prior_and_posterior(results, fwd_args, config, base_filename, t_obs,
                                                                    data, calibration_mode, output_index, save_plots,
                                                                    save_files)
    if save_files:
        outpath = os.path.join(os.path.split(os.getcwd())[0], 'output', base_filename + '_output.h5')
        save_results(results, fwd_args, config, outpath, data, mode='esmda')

        # try if we want to apply ACCC
        try:
            accc_timeinterval = config['ACCC_timeinterval']
            accc_timestart = config['ACCC_timestart']
            accc_step = config['ACCC_step']
            accc_maxstep = config['ACCC_maxstep']
            accc_step_sd = config['ACCC_step_sd']
            accc_low = config['ACCC_low']
            accc_slope = config['ACCC_slope']
            accc_scale = config['ACCC_scale']
            accc_cliplow = config['ACCC_cliplow']
            accc_cliphigh = config['ACCC_cliphigh']
            time_delay = config['time_delay']
            hammer_icu = config['hammer_ICU']
            hammer_slope = config['hammer_slope']
            hammer_release = config['hammer_release']
            hammer_alpha = config['hammer_alpha']
            accc_results = apply_accc(base_filename, results['M'][-1], fwd_args, accc_timeinterval,
                                      accc_timestart, accc_maxstep, accc_step, accc_step_sd, accc_low, accc_slope,
                                      accc_scale, hammer_icu, hammer_slope, hammer_alpha, hammer_release, accc_cliplow,
                                      accc_cliphigh, time_delay, data_end=data[-1, 0])
            # TODO: Add hammer results to h5 file and plotting
            results['fw'][-1] = accc_results
            save_and_plot_prior_and_posterior(results, fwd_args, config, base_filename, t_obs, data, calibration_mode,
                                              output_index, save_plots, save_files, plothammer=True)
            add_hammer_to_results(accc_results, outpath, mode='esmda')
        except KeyError:
            pass  # Don't apply ACCC

    return prior_and_posterior_results, results


def main(configpath):
    # Load the model configuration file and the data (observed cases)
    config = load_config(configpath)
    data = load_data(config)
    base_filename = (os.path.split(configpath)[-1]).split('.')[0]
    run_esmda_model(base_filename, config, data)


if __name__ == '__main__':
    # This script only accepts one input argument: the path to the configuration .json file
    main(sys.argv[1])
