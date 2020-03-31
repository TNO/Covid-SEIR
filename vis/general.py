from src.tools import generate_hos_actual
import numpy as np
import matplotlib.pyplot as plt

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


def plot_posterior(results, config, outbase, data, data_labels):
    calmodes = [S_HOS, S_ICU, S_HOSCUM, S_DEAD]
    o_indices = [O_HOS, O_ICU, O_HOSCUM, O_DEAD]

    useworldfile = config['worldfile']
    if (useworldfile):
        calmodes = [S_DEAD]
        o_indices = [O_DEAD]
    else:
        # check if I_HOS is present in the data, if not generate it
        data = generate_hos_actual(data, config)

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
            y_obs = np.full_like(data[:,0],fill_value=np.nan)
            title = ''

        t_obs = data[:,data_labels.index('time')]


        # Plot posterior
        posterior = results
        transparancy = max(min(5.0 / posterior.shape[0], 1),0.01)
        posterior_curves = np.array([member[output_index, :] for member in posterior]).T
        time = results[:,O_TIME,:]
        time_main = np.roll(time[0],int(time[0,0]))
        if time[0,0]!= 0.0:
            time_main[int(time[0,0]):]= np.nan
        # Shift time solutions
        posterior_shifted = np.zeros_like(posterior_curves)
        for i, post in enumerate(posterior_curves.T):
            shift = int(time[i,0])
            posterior_shifted[:,i] = np.roll(post,shift)
            if time[0, 0] != 0.0:
                posterior_shifted[shift:,i] = np.nan

        post_mean = np.mean(posterior_shifted, axis=-1)
        post_med = np.median(posterior_shifted, axis=-1)
        color = 'blue'
        plt.scatter(t_obs, y_obs, marker='o', c='k', label='Data')

        plt.plot(time_main, posterior_shifted, alpha=transparancy, c=color)
        plt.plot(time_main, post_mean, lw=2, ls=':', c='k', label='Mean of posterior')
        plt.plot(time_main, post_med, lw=2, c='k', label='Median of posterior')
        plt.xlim(0, config['XMAX'])
        plt.ylim(0, ymax)
        plt.xlabel('Time [days]')
        plt.ylabel('Number of cases')
        plt.legend(loc='upper left')
        plt.title(title + ' posterior ensemble')
        plt.savefig('{}posterior_ensemble_{}.png'.format(outbase, calmode), dpi=300)

        plt.xlim(0, time.max())
        plt.savefig('{}posterior_ensemble_{}_longterm.png'.format(outbase, calmode), dpi=300)
        plt.ylim(0, posterior_curves.max())
        plt.savefig('{}posterior_ensemble_{}_longterm_alt.png'.format(outbase, calmode), dpi=300)
        plt.close()

        p_values = config['p_values']
        p_array = []
        steps = np.arange(1, time.max() + 1)
        t_ind = [np.where(time_main == a)[0][0] for a in steps]
        posterior_length = posterior_shifted.shape[1]
        for post_day in posterior_shifted[t_ind, :]:
            array_sorted = np.sort(post_day)
            p_array.append([array_sorted[int(posterior_length * p)] for p in p_values])

        h_pvalues = ['P' + str(int(100 * a)) for a in p_values]
        header = 'time,mean,' + ','.join(h_pvalues) + ',observed'
        p_array = np.asarray(p_array)
        observed = np.pad(y_obs, (0, len(steps) - len(y_obs)), mode='constant', constant_values=np.nan)[:, None]
        table = np.concatenate((steps[:, None], post_mean[t_ind, None], p_array, observed), axis=1)
        np.savetxt('{}posterior_prob_{}_calibrated_on_{}.csv'.format(outbase, calmode, config['calibration_mode']),
                   table, header=header, delimiter=',', comments='', fmt='%.2f')