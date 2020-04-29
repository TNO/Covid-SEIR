import os
import warnings

from src.io_func import load_config, load_data
from src.tools import calc_axis_interval
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates

from src.parse import get_mean
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
O_CUMINF = 11
O_ALPHARUN =12

S_HOS = 'hospitalized'
S_ICU = 'ICU'
S_HOSCUM = 'hospitalizedcum'
S_DEAD = 'dead'
S_INF = 'infected'
S_ALPHARUN = 'alpharun'


def plot_confidence_alpha_obsolete(configpath, config, inputdata, firstdate):

    base = (os.path.split(configpath)[-1]).split('.')[0]
    outpath = os.path.join(os.path.split(os.getcwd())[0], 'output', base)

    names = ['posterior', 'prior']


    for i in range(0, 2):
        modelpath = '{}_{}_prob_{}_calibrated_on_{}.csv'.format(outpath, names[i], 'alpha', config['calibration_mode'])

        # modelpath = 'c:\\Users\\weesjdamv\\git\\corona\\configs\\r0_reductie_rivm.csv'
        # read the
        warnings.filterwarnings("error")
        try:
            modeldata = np.genfromtxt(modelpath, names=True, delimiter=',')
        except IOError as e:
            print('No alpha file found (rerun esmda), expecting alpha in:', modelpath)
            return

        xmax = config['XMAX']
        time_delay = config['time_delay']
        xmax = xmax + time_delay
        # read xmax
        try:
            xmax = config['plot']['xmaxalpha'] + time_delay
        except :
            print('No xmaxalpha in plot parameters, using XMAX:', xmax)
            pass

        casename = ''
        try:
            casename = config['plot']['casename']
        except:
            print('No casename in plot parameters')
            pass

        conf_level = [a for a in modeldata.dtype.names if 'P' in a]
        conf_range = float(conf_level[-1].strip('P')) - float(conf_level[0].strip('P'))
        time = modeldata['time']


        date_1 = datetime.datetime.strptime(firstdate, "%m/%d/%y")
        t = [date_1 + datetime.timedelta(days=a - 1) for a in time]
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        day_interval = calc_axis_interval((t[xmax] - t[0]).days)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))

        fig, ax = plt.subplots()
        try:
            figure_size = config['plot']['figure_size']
            assert len(figure_size) == 2
            plt.figure(figsize=figure_size)
        except:
            pass


        title = '1-$\\alpha$'
        symcolor = ['violet', 'purple']
        ls = ['-', '--']
        for ilevel, cl in enumerate(conf_level[2:-2]):
            plt.plot(t, 1-modeldata[cl], label=cl, c='k', ls=ls[ilevel], lw=0.25)
        for iconf in range(0, 2):
            conf_range = float(conf_level[-1-iconf].strip('P')) - float(conf_level[iconf].strip('P'))
            plt.fill_between(t, 1-modeldata[conf_level[iconf]], 1-modeldata[conf_level[-1-iconf]],
                             label='{}% confidence interval'.format(conf_range), color=symcolor[iconf])
        plt.grid(True)

        plt.legend(loc='lower left')
        plt.xlabel('Date')
        plt.ylabel('Number of cases')
        title = title + ' ' + casename
        plt.title(title)
        # plt.yscale('log')
        # plt.savefig('Hospital_cases_log.png', dpi=300)
        plt.yscale('linear')
        plt.xlim([date_1, t[xmax]])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        day_interval = calc_axis_interval((t[xmax] - t[0]).days)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
        plt.gcf().autofmt_xdate()
        plt.ylim(0, 1)
        outputpath = '{}_{}_prob_{}_calibrated_on_{}.png'.format(outpath, names[i], 'alpha', config['calibration_mode'])
        plt.savefig(outputpath, dpi=300)


def plot_confidence(outpath, config, inputdata, firstdate):

    calmodes = [S_HOS, S_ICU, S_HOSCUM, S_DEAD, S_INF, S_ALPHARUN]
    o_indices = [O_HOS, O_ICU, O_HOSCUM, O_DEAD, O_CUMINF, O_ALPHARUN]

    titles = ['Hospitalized', 'ICU', 'Hospitalized Cum.', 'Mortalities', 'Infected','1-$\\alpha$']
    y_label = ['Number of cases', 'Number of cases', 'Number of cases', 'Number of cases', 'Number of cases',
               '1-$\\alpha$']
    symcolors = [['powderblue','steelblue' ],['peachpuff', 'sandybrown'], ['lightgreen','forestgreen' ], ['silver', 'grey'], ['mistyrose', 'lightcoral'], ['plum', 'mediumorchid']]
    y_obs_s = [inputdata[:, I_HOS], inputdata[:, I_ICU], inputdata[:, I_HOSCUM], inputdata[:, I_DEAD], inputdata[:,I_INF], []]
    y_maxdef = config['YMAX']
    y_maxhos = y_maxdef * get_mean(config['hosfrac'])
    y_maxicu = y_maxhos * get_mean(config['ICufrac'])
    y_maxdead = y_maxhos * get_mean(config['dfrac']) * 4
    y_maxinf = y_maxdef*5
    y_max = [y_maxhos*1.5, y_maxicu, y_maxhos*4, y_maxdead, y_maxinf, 1]

    casename = ''
    try:
        casename = config['plot']['casename']
    except:
        print('No casename in plot parameters')
        pass

    daily = False
    try:
        daily = config['plot']['daily']
    except:
        print('No daily in plot parameters, assuming cumulative display of mortalities and hospitalized cum. Infected, ICU and hospitalized plotted as actual')
        pass

    x_obs = inputdata[:, 0]
    xmax = config['XMAX']
    time_delay = config['time_delay']
    xmax = xmax + time_delay
    #  make four output plots for hospitalized, cum hospitalized, dead, and hosm
    for i, calmode in enumerate(calmodes):
        output_index = o_indices[i]
        y_obs = None
        title = titles[i]
        symcolor = symcolors[i]
        y_obs = y_obs_s[i]
        ymax = y_max[i]
        ylabel = y_label[i]
        if (daily):
            if ((i==2) or (i==3) or (i==4)):
                y_obs = np.concatenate((np.array([0]), np.diff(y_obs)))
                title = 'Daily ' + titles[i]
                ymax = ymax *0.1

        # read the
        modelpath = '{}_posterior_prob_{}_calibrated_on_{}.csv'.format(outpath, calmode, config['calibration_mode'])
        modeldata = np.genfromtxt(modelpath, names=True, delimiter=',')

        conf_level = [a for a in modeldata.dtype.names if 'P' in a]
        if (daily):
            if ((i == 2) or (i == 3)or (i==4)):
                for ilevel, cl in enumerate(conf_level):
                    modeldata[cl] = np.concatenate((np.array([0]), np.diff(modeldata[cl])))
                modeldata['mean'] = np.concatenate((np.array([0]), np.diff(modeldata['mean'])))


        time = modeldata['time']
        mean = modeldata['mean']

        # fig, ax = plt.subplots()
        try:
            figure_size = config['plot']['figure_size']
            assert len(figure_size) == 2
            plt.figure(figsize=figure_size)
        except:
            plt.figure()
        date_1 = datetime.datetime.strptime(firstdate, "%m/%d/%y")
        t = [date_1 + datetime.timedelta(days=a - 1) for a in time]
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        day_interval = calc_axis_interval((t[xmax] - t[0]).days)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))

        plt.plot(t, mean, label='Mean (Expectation value)', c='k', lw=0.5)
        symcolor = symcolors[i]
        ls = ['-', '--']
        for ilevel, cl in enumerate(conf_level[2:-2]):
            plt.plot(t, modeldata[cl], label=cl, c='k', ls=ls[ilevel], lw=0.25)
        for iconf in range(0, 2):
            conf_range = float(conf_level[-1-iconf].strip('P')) - float(conf_level[iconf].strip('P'))
            c = symcolor[iconf]
            plt.fill_between(t, modeldata[conf_level[iconf]], modeldata[conf_level[-1-iconf]],
                             label='{}% confidence interval'.format(conf_range), color=c)
        if np.size(y_obs) > 0:
            x_days = [date_1 + datetime.timedelta(days=a - 1) for a in x_obs]
            plt.scatter(x_days, y_obs, c='k', label='data', marker='o', s=8)

        plt.grid(True)

        legendloc = 'upper left'
        try:
            legendloc = config['plot']['legendloc']
        except:
            print('No legendloc plot parameters, taking', legendloc)
            pass


        plt.legend(loc=legendloc)
        plt.xlabel('Date')
        plt.ylabel(ylabel)
        title = title + ' ' + casename
        plt.title(title)
        if (config['plot']['y_axis_log']):
            plt.yscale('log')
            if (ymax > 0):
                plt.ylim(1, ymax)
        else:
            plt.yscale('linear')
            plt.ylim(0, ymax)
        # plt.savefig('Hospital_cases_log.png', dpi=300)

        #plt.xlim([t[0], t[-1]])
        plt.xlim([t[0], t[xmax]])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        day_interval = calc_axis_interval((t[xmax] - t[0]).days)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
        plt.gcf().autofmt_xdate()



        outputpath = '{}_posterior_prob_{}_calibrated_on_{}.png'.format(outpath, calmode, config['calibration_mode'])
        plt.savefig(outputpath, dpi=300)

        if np.size(y_obs) > 0:
            # Have to make a new plot to change the size
            plt.figure(figsize=plt.rcParams["figure.figsize"])

            plt.plot(t, mean, label='Mean (Expectation value)', c='k', lw=0.5)
            for ilevel, cl in enumerate(conf_level[2:-2]):
                plt.plot(t, modeldata[cl], label=cl, c='k', ls=ls[ilevel], lw=0.25)
            for iconf in range(0, 2):
                conf_range = float(conf_level[-1 - iconf].strip('P')) - float(conf_level[iconf].strip('P'))
                c = symcolor[iconf]
                plt.fill_between(t, modeldata[conf_level[iconf]], modeldata[conf_level[-1 - iconf]],
                                 label='{}% confidence interval'.format(conf_range), color=c)
            if y_obs.any():
                x_days = [date_1 + datetime.timedelta(days=a - 1) for a in x_obs]
                plt.scatter(x_days, y_obs, c='k', label='data', marker='o', s=8)

            plt.grid(True)
            plt.xlabel('Date')
            plt.ylabel('Number of cases')
            plt.title(title)
            plt.yscale('linear')

            legendloc = 'lower right'
            try:
                legendloc = config['plot']['legendloczoom']
            except:
                print('No legendloczoom plot zoom parameters, taking', legendloc)
                pass
            plt.yscale('linear')
            plt.legend(loc=legendloc)
            inow = np.size(y_obs)
            i1 = inow-15
            i1 = max(0, i1)
            # ax.axvline(x=inow, color='silver')
            i2 = i1 + 25
            plt.xlim([t[i1], t[i2]])
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            day_interval = calc_axis_interval((t[i2] - t[i1]).days)
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
            plt.ylim(np.min(modeldata[conf_level[0]][i1:i2]), np.max(modeldata[conf_level[-1]][i1:i2]))
            outputpath = '{}_posterior_prob_{}_calibrated_on_{}_zoom.png'.format(outpath, calmode, config['calibration_mode'])
            plt.savefig(outputpath, dpi=300)
            plt.close()


def main(configpath):
    # Load the model configuration file and the data (observed cases)
    config = load_config(configpath)
    data = load_data(config)

    useworldfile = config['worldfile']
    if (not useworldfile):
        data = generate_hos_actual(data, config)
    else:
        data = generate_zero_columns(data, config)

    base = (os.path.split(configpath)[-1]).split('.')[0]
    outpath = os.path.join(os.path.split(os.getcwd())[0], 'output', base)

    plot_confidence(outpath, config, data, config['startdate'])
    #plot_confidence_alpha_obsolete(configpath, config, data, config['startdate'])


if __name__ == '__main__':
    # This script accepts two input argument:
    # 1) The path to the datafile to be postprocessed (.h5)
    # 2) the path to the configuration .json file
    main(sys.argv[1])
