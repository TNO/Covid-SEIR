import os
import warnings

from src.io_func import load_config, load_data
import sys
import numpy as np
import matplotlib.pyplot as plt

from src.tools import generate_hos_actual

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



def plot_confidence_alpha (configpath, config, inputdata):

    base = (os.path.split(configpath)[-1]).split('.')[0]
    outpath = os.path.join(os.path.split(os.getcwd())[0], 'output', base)
    modelpath = '{}_posterior_prob_{}_calibrated_on_{}.csv'.format(outpath, 'alpha', config['calibration_mode'])


   # modelpath = 'c:\\Users\\weesjdamv\\git\\corona\\configs\\r0_reductie_rivm.csv'
    # read the
    warnings.filterwarnings("error")
    try:
        modeldata = np.genfromtxt(modelpath, names=True, delimiter=',')
    except IOError as e:
        print ( 'no alpha file found (rerun esmda), expecting alpha in:', modelpath)
        return

    xmax  = config['XMAX']
    # read xmax
    try:
        xmax = config['plot']['xmaxalpha']
    except :
        print('no xmaxalpha in plot parameters, using XMAX:', xmax)
        pass

    casename = ''
    try:
        casename = config['plot']['casename']
    except:
        print('no casename in plot parameters')
        pass


    conf_level = [a for a in modeldata.dtype.names if 'P' in a]
    conf_range = float(conf_level[-1].strip('P')) - float(conf_level[0].strip('P'))
    t = modeldata['time']


    fig, ax = plt.subplots()
    # plt.figure()

    title = '1-alpha'  #'r$\alpha$'
    symcolor = 'silver'
    ls = [':', '-', '--', '-.']
    for i, cl in enumerate(conf_level[1:-1]):
        lw = 1.5 if cl == 'P50' else 0.5
        plt.plot(t, 1-modeldata[cl], label=cl, c='k', ls=ls[i], lw=lw)
    plt.fill_between(t, 1-modeldata[conf_level[0]], 1-modeldata[conf_level[-1]],
                     label='{}% confidence interval'.format(conf_range), color=symcolor)
    plt.grid(True)

    plt.legend(loc='lower left')
    plt.xlabel('Days')
    plt.ylabel('number of cases')
    title = title +' '+casename
    plt.title(title)
    # plt.yscale('log')
    # plt.savefig('Hospital_cases_log.png', dpi=300)
    plt.yscale('linear')
    plt.xlim(0, xmax)
    plt.ylim(0, 1)
    outputpath = '{}_posterior_prob_{}_calibrated_on_{}.png'.format(outpath, 'alpha', config['calibration_mode'])
    plt.savefig(outputpath, dpi=300)



def plot_confidence (configpath, config, inputdata):

    base = (os.path.split(configpath)[-1]).split('.')[0]
    outpath = os.path.join(os.path.split(os.getcwd())[0], 'output', base)
    calmodes = [S_HOS, S_ICU, S_HOSCUM, S_DEAD]
    o_indices = [O_HOS, O_ICU, O_HOSCUM, O_DEAD]

    useworldfile = config['worldfile']

    casename = ''
    try:
        casename = config['plot']['casename']
    except:
        print('no casename in plot parameters')
        pass

    x_obs = inputdata[:,0];
    xmax = config['XMAX']
    #  make four output plots for hospitalized, cum hospitalized, dead, and hosm
    for i, calmode in enumerate(calmodes):
        output_index = o_indices[i]
        y_obs = None
        if calmode == S_HOS:
            title = 'Hospitalized'
            ymax = config['YMAX'] * config['hosfrac']
            if not useworldfile:
                y_obs = inputdata[:, I_HOS]
            symcolor = 'lightskyblue'
        elif calmode == S_ICU:
            title = 'ICU'
            ymax = config['YMAX'] * config['hosfrac'] * config['ICufrac']
            if not useworldfile:
                y_obs = inputdata[:, I_ICU]
            symcolor = 'coral'
        elif calmode == S_HOSCUM:
            title = 'Hospitalized Cumulative'
            ymax = config['YMAX'] * config['hosfrac'] * 4
            if not useworldfile:
                y_obs = inputdata[:, I_HOSCUM]
            symcolor = 'aquamarine'
        elif calmode == S_DEAD:
            title = 'Mortalities'
            ymax = config['YMAX'] * config['hosfrac'] * config['dfrac'] * 4
            symcolor = 'lightgrey'
            if not useworldfile:
                y_obs = inputdata[:, I_DEAD]
        else:
            ymax = config['YMAX']

        # read the
        modelpath = '{}_posterior_prob_{}_calibrated_on_{}.csv'.format(outpath, calmode, config['calibration_mode'])
        modeldata = np.genfromtxt(modelpath, names=True, delimiter=',')
        conf_level = [a for a in modeldata.dtype.names if 'P' in a]
        conf_range = float(conf_level[-1].strip('P')) - float(conf_level[0].strip('P'))
        t = modeldata['time']
        mean = modeldata['mean']



     #   if (calmode==S_ICU):
     #       tbreakdown = 120
     #       corrbreakdown =  0.17 / 0.35
     #       tcorr = t*0 +1
     #       #[corrbreakdown if  i>tbreakdown else num for i, num in enumerate(tcorr)]
     #       for i, num in enumerate(tcorr):
     #          if (i>tbreakdown):
     #               tcorr[i] = corrbreakdown
     #       offset = mean[tbreakdown]
     #       mean = tcorr * (mean- offset) + offset
     #       for i, cl in enumerate(conf_level):
     #           offset = modeldata[cl][tbreakdown]
     #           modeldata[cl] = tcorr * (modeldata[cl] - offset) + offset


        fig, ax = plt.subplots()
        #plt.figure()

        plt.plot(t, mean, label='Mean (Expectation value)', c='k', lw=0.5)
        ls = [':', '-', '--', '-.']
        for i, cl in enumerate(conf_level[1:-1]):
            lw = 1.5 if cl=='P50' else 0.5
            plt.plot(t, modeldata[cl], label=cl, c='k', ls=ls[i], lw=lw)
        plt.fill_between(t, modeldata[conf_level[0]], modeldata[conf_level[-1]],
                         label='{}% confidence interval'.format(conf_range), color=symcolor)
        if y_obs.any():
           plt.scatter(x_obs, y_obs, c='k', label='data',marker='o',s=8)

        plt.grid(True)

        plt.legend(loc='upper left')
        plt.xlabel('Days')
        plt.ylabel('number of cases')
        title = title + ' ' + casename
        plt.title(title)
        #plt.yscale('log')
        #plt.savefig('Hospital_cases_log.png', dpi=300)
        plt.yscale('linear')
        plt.xlim([t[0],t[-1]])
        plt.ylim(0,ymax)
        outputpath =  '{}_posterior_prob_{}_calibrated_on_{}.png'.format(outpath, calmode, config['calibration_mode'])
        plt.savefig(outputpath, dpi=300)

        if y_obs.any():
            plt.legend(loc='lower right')
            inow = np.size(y_obs)
            i1 = inow-15
            i1 = max(0,i1)
            ax.axvline(x=inow, color='silver')
            i2 = i1 + 25
            plt.xlim([t[i1], t[i2]])
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

    plot_confidence(configpath, config, data)
    plot_confidence_alpha(configpath, config, data)


if __name__ == '__main__':
    # This script accepts two input argument:
    # 1) The path to the datafile to be postprocessed (.h5)
    # 2) the path to the configuration .json file
    main(sys.argv[1])
