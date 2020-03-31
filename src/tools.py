import numpy as np

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
    delay_rec = config['delayREC']
    delay_hos = config['delayHOS']
    delay_hosrec =config['delayHOSREC']
    delay_hosd = config['delayHOSD']
    delay_icu =config['delayICUCAND']
    delay_icud = config['delayICUD']
    delay_icurec = config['delayICUREC']
    hosfrac =config['hosfrac']
    dfrac = config['dfrac']
    icufrac =config['ICufrac']
    icudfrac = config['icudfrac']

    hoscum = data[:, I_HOSCUM]

    dpfrac = (dfrac - icufrac * icudfrac) / (1 - icufrac * icudfrac)
    icucum = np.roll(hoscum, int(delay_icu))*icufrac
    icucum[:int(delay_icu)] = 0
    icu_dead = np.roll(icucum, int(delay_icud)) * icudfrac
    icu_dead[: int(delay_icud)] = 0
    icu_rechos = np.roll(icucum, int(delay_icurec)) * (1. - icudfrac) # Back to hospital
    icu_rechos[:int(delay_icurec)] = 0
    icu_recfull = np.roll(icu_rechos, int(delay_hosrec)) # All who go back to hospital recover, but this takes time
    # this is the full recovered patients who went through ICU
    icu_recfull[:int(delay_hosrec)] = 0

    # this is the full rovered patients who did not went through ICU
    rechos = np.roll(hoscum, int(delay_hosrec)) * (1. - dpfrac) * (1-icufrac)
    rechos[:int(delay_hosrec)] = 0

    hdead = np.roll(hoscum, int(delay_hosd)) * dpfrac
    hdead[:int(delay_hosd)] = 0

    dead = hdead + icu_dead
    hos = hoscum - rechos - icu_recfull - dead # includes ICU count

    hos = hos[:, None]
    datanew = np.concatenate((data[:,0:6], hos), axis=-1)
    return datanew