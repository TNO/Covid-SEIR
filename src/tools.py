import numpy as np

from src.io_func import read_icufrac_data
from src.parse import get_mean

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


def generate_hos_actual(data, config):
    # generate an additional data column for actual hospitalized
    # only do this when the mean of the actuals are zero
    delay_rec = config['delayREC']
    delay_hos = config['delayHOS']
    delay_hosrec =config['delayHOSREC']
    delay_hosd = config['delayHOSD']
    delay_icu =config['delayICUCAND']
    delay_icud = config['delayICUD']
    delay_icurec = config['delayICUREC']
    hosfrac =config['hosfrac']
    dfrac = config['dfrac']
    t_max = np.size(data[:, I_HOSCUM])
    time = np.linspace(1, t_max, int(t_max) )
    icufrac = read_icufrac_data(config, time,0)
    # icufrac =config['ICufrac']
    icudfrac = config['icudfrac']

    hos = data[:,I_HOS]

    if np.sum(np.cumsum(hos))>0.1:
        return data

    # else include the hospitalised column

    hoscum = data[:, I_HOSCUM]

    # icufrac is the fraction of patients going to ICU at the time of entry in hospital
    icufrac2 = icufrac
    # dp frac is the fraction of all  patients dying who did not go to ICU, referenced at entry in the hospital
    dpfrac = (dfrac - icufrac * icudfrac) / (1.0001 - icufrac)  # / (1 - icufrac * icudfrac)
    hosday2 = np.concatenate((np.array([0]), np.diff(hoscum)))

    hosday = hosday2 *0.0



    if (delay_icu>-0.01):
        for i, num in enumerate(hosday):
            irange = int(delay_icu*2)
            ilow =int (max(0,(i-irange)))
            mn = np.mean( hosday2[ilow:i+1])
            hosday[i] = mn
    # construct hospitalization and icu as if nobody dies in the hospital, except icu
    icuday = hosday * icufrac
    icucum = np.cumsum(icuday)
    icucum = np.roll(icucum, int(delay_icu))
    icucum[:int(delay_icu)] = 0

    # recovered from icu, taking into account deaths from icu

    idelay_icurec = get_mean(delay_icurec)
    icu_rechos = np.roll(icucum, int(idelay_icurec)) * (1. - icudfrac)  # Back to hospital
    icu_rechos[:int(idelay_icurec)] = 0
    idelay_hosrec = get_mean(delay_hosrec)
    icu_recfull = np.roll(icu_rechos, int(idelay_hosrec))  # All who go back to hospital recover, but this takes time
    # this is the full recovered patients who went through ICU
    icu_recfull[:int(idelay_hosrec)] = 0

    idelay_icud = get_mean(delay_icud)
    # dead from icu
    icu_dead = np.roll(icucum, int(idelay_icud)) * icudfrac
    icu_dead[: int(idelay_icud)] = 0

    idelay_hosrec = get_mean(delay_hosrec)
    # this is the full rovered patients who did not went through ICU
    rechos = np.roll(hoscum * (1 - icufrac) * (1. - dpfrac), int(idelay_hosrec))
    rechos[:int(idelay_hosrec)] = 0
    # dead from hospitalized but not including icu
    idelay_hosd = get_mean(delay_hosd)
    hdead = np.roll(hoscum * (1 - icufrac) * dpfrac, int(idelay_hosd))
    hdead[:int(idelay_hosd)] = 0



    # actual icu is cumulative icu minus icu dead and icu revovered
    icu = icucum - icu_dead - icu_rechos
    dead = hdead + icu_dead
    hos = hoscum - rechos - icu_recfull - dead  # includes ICU count


    hos = hos[:, None]
    datanew = np.concatenate((data[:,0:6], hos), axis=-1)
    return datanew