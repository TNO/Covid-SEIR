import os
import warnings

from src.io_func import load_config, load_data, read_icufrac_data
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates

from src.parse import get_mean, to_gauss_smooth_sd
from src.tools import gauss_smooth_shift, do_hospitalization_process, calc_axis_interval

I_ICUCUM = 0
I_ICUDEAD = 1
I_ICUREC =2
I_ICUPRES =3
I_HOSCUM = 5

S_ICUCUM = 'Cumulative ICU'
S_ICUDEAD = 'Mortalities ICU'
S_ICUREC ='Recovered ICU'
S_ICUPRES ='Present ICU'
S_HOSPREC = 'Recovered ICU'
S_HOSCUM = 'Cumulative Hospitalized'

IP_DELAYREC = 0
IP_DELAYHOS = 1
IP_DELAYHOSREC = 2
IP_DELAYHOSD = 3
IP_DELAYICU = 4
IP_DELAYICUD = 5
IP_DELAYICUREC = 6

IP_HOSFRAC = 0
IP_DFRAC = 1
IP_ICUFRAC = 2
IP_ICUDFRAC = 3


def save_input_data(filename, time, icufrac, tmin, tmax):
    i1 = int(np.where(time==tmin)[0])
    i2 = int(np.where(time==tmax)[0])
    timeclip = time[i1:i2+1]
    icufracclip = icufrac[i1:i2+1]

    header = 'day icufrac'
   # table = datanew[:, 0:7]  # np.concatenate((datanew[:, 0:6]), axis=-1)
    table = np.concatenate((timeclip[:, None], icufracclip[:, None]), axis=-1)
    np.savetxt(filename, table, header=header, delimiter=' ', comments='', fmt='%8f')


def save_and_plot_IC(config, output_base_filename, save_plots):
    outpath = os.path.join(os.path.split(os.getcwd())[0], 'output', output_base_filename)

    firstdate = config['startdate']

    direc = '.'
    # read icufracs from a datafile, these are
    try:
        ic_data_file = config['icdatafile']
    except KeyError:
        print("No icdatafile in config, using '../res/icdata_main.txt'.")
        ic_data_file = '../res/icdata_main.txt'
    data = np.genfromtxt(os.path.join(direc, ic_data_file), names=True)

    # Check if database has RIVM hospitalized data. If NOT, then delete that row
    if data[-1][-1] == 0:
        data = data[:-1]

    t = data['day']
    tmin = t[0]
    tmax = t[-1]
    icucum_obs = data['icucum']
    icudead_obs = data['icudead']
    icurec_obs = data['icurec']
    icupres_obs = data['icupres']
    hosprec_obs = data['hosprec']
    hoscum_obs = data['hoscum']

    delay_hosrec = 0

    delay_rec = get_mean(config['delayREC'])
    delay_hos = get_mean(config['delayHOS'])
    #delay_hosrec = get_mean(config['delayHOSREC'])
    delay_hosd = get_mean(config['delayHOSD'])
    delay_icu = get_mean(config['delayICUCAND'])
    delay_icud = get_mean(config['delayICUD'])
    delay_icurec = get_mean(config['delayICUREC'])
    hosfrac = get_mean(config['hosfrac'])
    dfrac = get_mean(config['dfrac'])

    rec_sd = to_gauss_smooth_sd(config['delayREC'])
    hos_sd = to_gauss_smooth_sd(config['delayHOS'])
    hosrec_sd = to_gauss_smooth_sd(config['delayHOSREC'])
    hosd_sd = to_gauss_smooth_sd(config['delayHOSD'])

    icu_sd = to_gauss_smooth_sd(config['delayICUCAND'])
    icud_sd = to_gauss_smooth_sd(config['delayICUD'])
    icurec_sd = to_gauss_smooth_sd(config['delayICUREC'])

    icufrac = 0.35
    # icufrac =config['ICufrac']
    icudfrac = get_mean(config['icudfrac'])

    names = [S_ICUCUM, S_ICUDEAD, S_HOSPREC, S_ICUPRES,S_ICUREC,  S_HOSCUM]
    #y_obs = [icucum_obs, icudead_obs, icurec_obs, icupres_obs, hosprec_obs]
    y_obs = [icucum_obs, icudead_obs, hosprec_obs+ icurec_obs, icupres_obs, icurec_obs, hoscum_obs]
    colors = ['maroon', 'black', 'darkgreen','coral','mistyrose', 'lightgreen']
    colors = ['royalblue', 'black', 'darkgreen','sandybrown','mistyrose', 'coral']



    nstart = 10
    tstart = np.arange(-10,1,1)
    other0 = np.zeros(len(tstart))
    t = np.concatenate((tstart, t), axis=-1)
    for i, y in enumerate(y_obs):
        y_obs[i] = np.concatenate((other0, y), axis=-1)

    icurate = 15 # 15
    hosrate = 80
    nend = 20
    tend= np.arange(tmax+1,tmax+nend,1)
    icugrowth = (tend - tend[0] + 1)*icurate
    hosgrowth = (tend - tend[0] + 1) * hosrate
    other0 = np.zeros(len(tend))
    t=  np.concatenate(( t, tend), axis=-1)
    for i, y in enumerate(y_obs):
        other = other0+y[-1]
        if (i==I_ICUCUM):
            other = other + icugrowth
        if (i == I_HOSCUM):
            other = other + hosgrowth
        y_obs[i] = np.concatenate((y, other), axis=-1)

    icucum_obs = y_obs[0]
    icudead_obs = y_obs[1]
    icurec_obs = y_obs[2]
    icupres_obs = y_obs[3]
    hosprec_obs = y_obs[4]
    hoscum_obs = y_obs[5]


    gausssmooth_hosp =1.5
    gausssmooth_icu= 1.5
    germanmax =0

    hoscum_pred = hoscum_obs
    hoscum_pred = gauss_smooth_shift(hoscum_obs, 0, gausssmooth_hosp)
    # estimate icu frac

    icucum = icucum_obs * 1.0
    icucorrectgerman = icucum *0.0
    germandaily = 8.0
    indexscale = np.asarray(np.where(t>30)[0])

    for i, index  in enumerate(indexscale):
        ioffset = indexscale[0]
        iact = index
        icucorrectgerman[iact] = min(germanmax,(iact-ioffset)*germandaily)
    icucum = icucum_obs - icucorrectgerman

    #hos = dataprocess[:, OP_HOS]

    date_1 = datetime.datetime.strptime(firstdate, "%m/%d/%y")
    tplot = [date_1 + datetime.timedelta(days=a - 1) for a in t]

    dodirect = False
    if (dodirect):
        totalremoved = hosprec_obs + icurec_obs+icudead_obs
        exp_pres = icucum_obs-totalremoved
        obs_pres = icupres_obs
        dif = exp_pres-obs_pres


        icucum = np.roll(icucum, int(delay_icu))
        icucum[:int(delay_icu)] = 0

        icu_rechos= gauss_smooth_shift(icucum, delay_icurec, icurec_sd, scale=(1-icudfrac))
        icu_recfull = gauss_smooth_shift(icu_rechos, delay_hosrec, 0)
        # dead from icu
        icu_dead = gauss_smooth_shift(icucum, delay_icud, icud_sd, scale=icudfrac)
        icu = icucum - icu_dead - icu_rechos
    else:
        OP_HOS = 0
        OP_HOSCUM = 1
        OP_ICU = 2
        OP_ICUCUM = 3
        OP_REC = 4
        OP_DEAD = 5
        OP_ICUREC = 6
        OP_ICUDEAD = 7

        hosday2 = np.concatenate((np.array([0]), np.diff(hoscum_pred)))
        hosday = hosday2 * 1.0
        hosday = np.clip(hosday,1,max(hosday))
        icuday2 = np.concatenate((np.array([0]), np.diff(icucum)))
        icuday = icuday2*1.0
        icuday = gauss_smooth_shift(icuday, 0, gausssmooth_icu)
        icufrac = icuday/hosday
        icufrac= np.clip(icufrac,0,1)

        # Check if the icufrac file is defined in config, otherwise use the output_base_filename
        try:
            filename = os.path.split(config['icufracfile'])[-1]
            icufrac_filename = os.path.join(os.path.split(os.getcwd())[0], 'output', filename)
        except KeyError:
            print("No icufracfile in config, using base filename {}_icufrac.txt".format(output_base_filename))
            filename = os.path.join(os.path.split(os.getcwd())[0], 'output', output_base_filename)
            icufrac_filename = '{}_icufrac.txt'.format(filename)

        save_input_data(icufrac_filename, t, icufrac, tmin, tmax)

        if save_plots:
            fracs = [hosfrac, dfrac, icufrac, icudfrac]
            delays = [delay_rec, delay_hos, delay_hosrec, delay_hosd, delay_icu, delay_icud, delay_icurec]
            gauss_sd = [rec_sd, hos_sd, hosrec_sd, hosd_sd, icu_sd, icud_sd, icurec_sd]
            #gauss_sd = [0, 0, 0, 0, 0, icud_sd, icurec_sd]

            r= gauss_smooth_shift(hoscum_pred, -delay_hos, hos_sd, scale=1.0/hosfrac)
            dataprocess = do_hospitalization_process(hoscum_pred, delays, fracs, gauss_stddev=gauss_sd, removed=r)

            rec = dataprocess[:, OP_REC, None]
            hos = dataprocess[:, OP_HOS, None]
            icu = dataprocess[:, OP_ICU, None]
            icucum = dataprocess[:, OP_ICUCUM, None]
            icu_dead = dataprocess[:, OP_ICUDEAD, None]
            icu_recfull = dataprocess[:,OP_ICUREC, None]

            fig, ax = plt.subplots()
            plt.figure(figsize=(6.0, 6.0))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            day_interval = calc_axis_interval((tplot[nstart] - tplot[-nend]).days)
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
            plt.gcf().autofmt_xdate()

            plt.legend(loc='upper left')
            plt.xlabel('Date',fontsize=12)
            plt.ylabel('number of cases',fontsize=12)
            width = 0.4
            p1 = plt.bar(tplot, icuday, width)
            p2 = plt.bar(tplot, (hosday-icuday), width, bottom=icuday)
            outputpath = '{}_{}_{}.png'.format(outpath, 'ICanalysis', 'hosICU')
            #plt.title('daily hospitalization and ICU')
            plt.legend((p1[0], p2[0]), ('Daily ICU', 'Daily Hospitalized'))
            plt.xlim([date_1, tplot[-nend]])
           # plt.xlim(tmin, tmax)
            plt.grid(True)
            plt.savefig(outputpath, dpi=300)

            fig, ax = plt.subplots()
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            day_interval = calc_axis_interval((tplot[nstart] - tplot[-nend]).days)
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
            plt.gcf().autofmt_xdate()
            plt.xlabel('Date')
            plt.title('daily intake ICU')
            plt.ylabel('fraction of daily hospitalization')
            width = 0.4
            rel = np.clip(icuday/hosday,0,1)
            one = rel*0.0 +1.0
            p1 = plt.bar(tplot, rel, width)
            #p2 = plt.bar(t, (one-rel), width, bottom=rel)
            outputpath = '{}_{}_{}.png'.format(outpath, 'ICanalysis', 'hosICU%')
            plt.xlim([date_1, tplot[-nend]])
            plt.grid(True)

            plt.savefig(outputpath, dpi=300)
            # plt.show()

    if save_plots:
        y_pred = [icucum, icu_dead, icu_recfull, icu, icu_recfull, hoscum_pred ]

        fig, ax = plt.subplots()
        plt.figure(figsize=(6.0, 6.0))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        day_interval = calc_axis_interval((tplot[nstart] - tplot[-nend]).days)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=day_interval))
        plt.gcf().autofmt_xdate()

        for i, name in enumerate(names):
            name2 = name + ' - data'
            if (i!=4):
                #plt.scatter(tplot, y_obs[i], c=colors[i], label=name2, marker='o', s=8)
                plt.scatter(tplot, y_obs[i], c=colors[i], label=name, marker='o', s=8)
            name2 = name + ' - model'
            if (i!=4):
                #plt.plot(tplot, y_pred[i],  c=colors[i], label=name2,  lw=2)
                plt.plot(tplot, y_pred[i], c=colors[i],  lw=2)
        #plt.scatter(t, exp_pres, c='b', label='icu pres (expected)', marker='o', s=8)

        plt.legend(loc='upper left')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('number of cases', fontsize=12)
        title = 'ICU analysis'
        #plt.title(title)
        # plt.yscale('log')
        # plt.savefig('Hospital_cases_log.png', dpi=300)
        plt.yscale('linear')
        plt.xlim([date_1, tplot[-nend]])
        plt.ylim(0, 3000)
        plt.grid(True)

        #plt.show()

        outputpath = '{}_{}_{}.png'.format(outpath,  'ICanalysis', 'data')
        plt.savefig(outputpath, dpi=300)


def main(configpath, save_plots=True):
    # Load the model configuration file and the data (observed cases)
    config = load_config(configpath)
    output_base_filename = (os.path.split(configpath)[-1]).split('.')[0]
    save_and_plot_IC(config, output_base_filename, save_plots)


if __name__ == '__main__':
    # This script accepts two input arguments:
    # 1) The path to the datafile to be postprocessed
    # 2) If you want to save the plots (defaults to True)
    main(sys.argv[1])
