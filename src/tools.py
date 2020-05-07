import numpy as np
from scipy.ndimage import gaussian_filter1d
import scipy.special as sps

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

OP_HOS = 0
OP_HOSCUM = 1
OP_ICU = 2
OP_ICUCUM = 3
OP_REC = 4
OP_DEAD = 5
OP_ICUREC = 6
OP_ICUDEAD = 7


def gauss_smooth_shift_OLD(input,  shift, stddev, scale=1.0):
    """
    smooths the input with gaussian smooothing with standarddeviation and shifts its delay positions
    :param input: The input array
    :param shift: the amount of indices to shift the result
    :param the stddev for the gaussian smoothing (in index count)
    :param scale: scale the input array first with scale
    :return: the smoothed and shifted array
    """
    forcescale = False
    if isinstance(scale, np.ndarray):
        forcescale = True
    if (forcescale or np.abs(scale-1) > 1e-5):
        input = input*scale

    result = input
    if (stddev > 0.0):
        result = gaussian_filter1d(input, stddev,  mode='nearest')

    result = np.roll(result, int(shift))
    if (shift > 0):
        result[: int(shift)] = 0
    #else:
        # backward roll can simply use the trailing values
    return result

def gaussian( x , s):
    return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )




def gamma_equivalent( x ,shape,scale ):
    if (x<0):
        return 0
    else:
        return (x**(shape-1) *(np.exp(-x/scale)/ (sps.gamma(shape)*scale**shape)))

def gamma_smooth_shift_convolve (input,  shift, stddev, scale=1.0):
    """
    smooths the input with gaussian smooothing with standarddeviation and shifts its delay positions
    :param input: The input array
    :param shift: the amount of indices to shift the result
    :param the stddev for the gaussian smoothing (in index count)
    :param scale: scale the input array first with scale
    :return: the smoothed and shifted array
    """
    forcescale = False
    if isinstance(scale, np.ndarray):
        forcescale = True
    if (forcescale or np.abs(scale-1) > 1e-5):
        input = input*scale

    result = input
    if (stddev > 0.99) and (shift>0.99):
        if (shift<stddev):
            stddev = shift-1
        isd = 3* max(1,int (stddev))
        isd = min( int(0.5*np.size(input)-1), isd)
        theta = stddev**2/shift
        k = shift/theta
        ishift = int (shift)
        mygamma= np.fromiter(( gamma_equivalent(x, k, theta) for x in range (-isd+ishift, isd+ishift+1)), np.float)
        result = np.convolve(input, mygamma, mode='same')

    if (shift > 0):
        result = np.roll(result, int(shift))
        result[: int(shift)] = 0
    #else:
        # backward roll can simply use the trailing values
    return result

def gauss_smooth_shift_convolve(input,  shift, stddev, scale=1.0):
    """
    smooths the input with gaussian smooothing with standarddeviation and shifts its delay positions
    :param input: The input array
    :param shift: the amount of indices to shift the result
    :param the stddev for the gaussian smoothing (in index count)
    :param scale: scale the input array first with scale
    :return: the smoothed and shifted array
    """
    forcescale = False
    if isinstance(scale, np.ndarray):
        forcescale = True
    if (forcescale or np.abs(scale-1) > 1e-5):
        input = input*scale

    result = input
    if (stddev > 0.0):
        isd = max(1,int (stddev))
        myGaussian = np.fromiter(( gaussian(x,stddev) for x in range (-3*isd,3*isd+1)), np.float)
        result = np.convolve(input, myGaussian, mode='same')

    result = np.roll(result, int(shift))
    if (shift > 0):
        result[: int(shift)] = 0
    #else:
        # backward roll can simply use the trailing values
    return result


def gauss_smooth_shift(input,  shift, stddev, scale=1.0):
    """
    smooths the input with gaussian smooothing with standarddeviation and shifts its delay positions
    :param input: The input array
    :param shift: the amount of indices to shift the result
    :param the stddev for the gaussian smoothing (in index count)
    :param scale: scale the input array first with scale
    :return: the smoothed and shifted array
    """
    forcescale = False
    if isinstance(scale, np.ndarray):
        forcescale = True
    if (forcescale or np.abs(scale-1) > 1e-5):
        input = input*scale

    result = input
    if (stddev > 0.0):
        result = gaussian_filter1d(input, stddev,  mode='nearest')


    result = np.roll(result, int(shift))
    if (shift > 0):
        result[: int(shift)] = 0
    #else:
        # backward roll can simply use the trailing values
    return result



def do_hospitalization_process(hoscum, delays, fracs, gauss_stddev=None, removed=None):

    icudfrac = fracs[IP_ICUDFRAC]
    dfrac =  fracs[IP_DFRAC]
    icufrac =fracs[IP_ICUFRAC]
    hosfrac = fracs[IP_HOSFRAC]

    delay_rec = delays[IP_DELAYREC]
    delay_hos = delays[IP_DELAYHOS]
    delay_hosrec =delays[IP_DELAYHOSREC]
    delay_hosd = delays[IP_DELAYHOSD]
    delay_icu =delays[IP_DELAYICU]
    delay_icud = delays[IP_DELAYICUD]
    delay_icurec =delays[IP_DELAYICUREC]

    gauss_sd = [0,0,0,0,0,0,0]
    if isinstance(gauss_stddev, list):
        gauss_sd = gauss_stddev

    rec_sd =  gauss_sd[IP_DELAYREC];
    hos_sd =  gauss_sd[IP_DELAYHOS]
    hosrec_sd =gauss_sd[IP_DELAYHOSREC]
    hosd_sd = gauss_sd[IP_DELAYHOSD]
    icu_sd =gauss_sd[IP_DELAYICU]
    icud_sd = gauss_sd[IP_DELAYICUD]
    icurec_sd =gauss_sd[IP_DELAYICUREC]

    # icufrac is the fraction of patients going to ICU at the time of entry in hospital
    icufrac2 = icufrac
    # dp frac is the fraction of all  patients dying who did not go to ICU, referenced at entry in the hospital
    dpfrac = (dfrac - icufrac * icudfrac) / (1.0001 - icufrac)  # / (1 - icufrac * icudfrac)
    hosday2 = np.concatenate((np.array([0]), np.diff(hoscum)))

    hosday = hosday2 * 1.0

    if (delay_icu > 0.01):
        for i, num in enumerate(hosday):
            irange = int(delay_icu*2)
            ilow = int(max(0,(i-irange)))
            mn = np.mean(hosday2[ilow:i+1])
            hosday[i] = mn
    # construct hospitalization and icu as if nobody dies in the hospital, except icu
    icuday = hosday * icufrac
    icucum = np.cumsum(icuday)



    stddev_norm = 0
    scale_norm =1
    icucum = gauss_smooth_shift_convolve(icucum, delay_icu, icu_sd, scale_norm)

   # icudayrec = gauss_smooth_shift_convolve(icuday, delay_icurec, icurec_sd, (1-icudfrac))
   # icurechos = np.cumsum(icudayrec)

    icu_rechos = gamma_smooth_shift_convolve(icucum, delay_icurec, icurec_sd, (1-icudfrac))
    icu_recfull = gauss_smooth_shift_convolve(icu_rechos, delay_hosrec, hosrec_sd, scale_norm)
    icu_dead = gamma_smooth_shift_convolve(icucum, delay_icud, icud_sd, icudfrac)

    rechos = gauss_smooth_shift_convolve(hoscum, delay_hosrec, hosrec_sd, (1-icufrac)*(1-dpfrac))
    hdead = gauss_smooth_shift_convolve(hoscum, delay_hosd, hosd_sd, (1-icufrac)*dpfrac)

    # actual icu is cumulative icu minus icu dead and icu revovered
    icu = icucum - icu_dead - icu_rechos
    dead = hdead + icu_dead
    hos = hoscum - rechos - icu_recfull - dead  # includes ICU count

    r = removed
    if not isinstance(r, np.ndarray):
        r=  gauss_smooth_shift(hoscum, -delay_hos, 0, scale=1.0/(hosfrac))

    recmild = gauss_smooth_shift(r, delay_rec, rec_sd, (1-hosfrac))

    rec = rechos + recmild + icu_recfull

    rec = rec[:, None]
    hos = hos[:, None]
    icu = icu[:, None]
    icucum = icucum[:,None]
    hoscum = hoscum[:, None]
    dead = dead[:, None]
    icurec = icu_rechos[:,None]
    icudead = icu_dead[:, None]

    resout= np.concatenate((hos,hoscum,icu,icucum,rec,dead, icurec, icudead), axis=-1)
    return resout


def do_hospitalization_process_OLD(hoscum, delays, fracs, gauss_stddev=None, removed=None):

    icudfrac = fracs[IP_ICUDFRAC]
    dfrac =  fracs[IP_DFRAC]
    icufrac =fracs[IP_ICUFRAC]
    hosfrac = fracs[IP_HOSFRAC]

    delay_rec = delays[IP_DELAYREC]
    delay_hos = delays[IP_DELAYHOS]
    delay_hosrec =delays[IP_DELAYHOSREC]
    delay_hosd = delays[IP_DELAYHOSD]
    delay_icu =delays[IP_DELAYICU]
    delay_icud = delays[IP_DELAYICUD]
    delay_icurec =delays[IP_DELAYICUREC]

    gauss_sd = [0,0,0,0,0,0,0]
    if isinstance(gauss_stddev, list):
        gauss_sd = gauss_stddev

    rec_sd =  gauss_sd[IP_DELAYREC];
    hos_sd =  gauss_sd[IP_DELAYHOS]
    hosrec_sd =gauss_sd[IP_DELAYHOSREC]
    hosd_sd = gauss_sd[IP_DELAYHOSD]
    icu_sd =gauss_sd[IP_DELAYICU]
    icud_sd = gauss_sd[IP_DELAYICUD]
    icurec_sd =gauss_sd[IP_DELAYICUREC]

    # icufrac is the fraction of patients going to ICU at the time of entry in hospital
    icufrac2 = icufrac
    # dp frac is the fraction of all  patients dying who did not go to ICU, referenced at entry in the hospital
    dpfrac = (dfrac - icufrac * icudfrac) / (1.0001 - icufrac)  # / (1 - icufrac * icudfrac)
    hosday2 = np.concatenate((np.array([0]), np.diff(hoscum)))

    hosday = hosday2 * 1.0

    if (delay_icu > 0.01):
        for i, num in enumerate(hosday):
            irange = int(delay_icu*2)
            ilow = int(max(0,(i-irange)))
            mn = np.mean(hosday2[ilow:i+1])
            hosday[i] = mn
    # construct hospitalization and icu as if nobody dies in the hospital, except icu
    icuday = hosday * icufrac
    icucum = np.cumsum(icuday)



    stddev_norm = 0
    scale_norm =1
    icucum = gauss_smooth_shift(icucum, delay_icu, icu_sd, scale_norm)


    icu_rechos = gauss_smooth_shift(icucum, delay_icurec, icurec_sd, (1-icudfrac))
    icu_recfull = gauss_smooth_shift(icu_rechos, delay_hosrec, hosrec_sd, scale_norm)
    icu_dead = gauss_smooth_shift(icucum, delay_icud, icud_sd, icudfrac)

    rechos = gauss_smooth_shift(hoscum, delay_hosrec, hosrec_sd, (1-icufrac)*(1-dpfrac))
    hdead = gauss_smooth_shift(hoscum, delay_hosd, hosd_sd, (1-icufrac)*dpfrac)

    # actual icu is cumulative icu minus icu dead and icu revovered
    icu = icucum - icu_dead - icu_rechos
    dead = hdead + icu_dead
    hos = hoscum - rechos - icu_recfull - dead  # includes ICU count

    r = removed
    if not isinstance(r, np.ndarray):
        r=  gauss_smooth_shift(hoscum, -delay_hos, 0, scale=1.0/(hosfrac))

    recmild = gauss_smooth_shift(r, delay_rec, rec_sd, (1-hosfrac))

    rec = rechos + recmild + icu_recfull

    rec = rec[:, None]
    hos = hos[:, None]
    icu = icu[:, None]
    icucum = icucum[:,None]
    hoscum = hoscum[:, None]
    dead = dead[:, None]
    icurec = icu_rechos[:,None]
    icudead = icu_dead[:, None]

    resout= np.concatenate((hos,hoscum,icu,icucum,rec,dead, icurec, icudead), axis=-1)
    return resout


def generate_zero_columns(data, config):
    # generate 3 additional data column (0 is first column index_
    # 4 hospitalized cumulative,
    # 5 ICU
    # 6
    days = data[:,I_TIME]
    tests = data[:, I_INF]
    dead = data[:, I_DEAD]
    #n = np.ndarray.size(days)
    delay_hos = get_mean(config['delayHOS'])
    delay_hos2 = delay_hos + 1/get_mean(config['gamma'])
    delay_hosd = get_mean(config['delayHOSD'])
    icufrac = read_icufrac_data(config, days,0)
    icudfrac = get_mean(config['icudfrac'])
    dfrac = get_mean(config['dfrac'])
    delay_icud = get_mean(config['delayICUD'])

    hoscum = days*0.0
    hoscumcreate = 'none'
    try:
        hoscumcreate = config['hoscumcreate']
    except:
        pass

    if (hoscumcreate=='dead'):
        hoscum1 = (1-icufrac)*gauss_smooth_shift(dead, delay_hosd, 0, 1.0 / (dfrac-icufrac*icudfrac))
        hoscum2 = (icufrac) *gauss_smooth_shift(dead, delay_icud, 0, 1.0 / ( icudfrac))
        hoscum = hoscum1 + hoscum2
    elif (hoscumcreate=='confirmed'):
        try:
            rate_hoscumfromconfirmed = config['rate_hoscumfromconfirmed']
        except:
            pass
        hoscum = gauss_smooth_shift(tests, delay_hos2, 0, rate_hoscumfromconfirmed)

    icu = days*0.0
    hosact = days*0.0
    datanew = np.concatenate((data[:,0:4], hoscum[:,None], icu[:,None], hosact[:,None]), axis=-1)
    datanew = generate_hos_actual(datanew , config)
    return datanew


def generate_hos_actual(data, config):
    # generate an additional data column for actual hospitalized
    # only do this when the mean of the actuals are zero

    delay_rec = get_mean(config['delayREC'])
    delay_hos = get_mean(config['delayHOS'])
    delay_hosrec =get_mean(config['delayHOSREC'])
    delay_hosd = get_mean(config['delayHOSD'])
    delay_icu =get_mean(config['delayICUCAND'])
    delay_icud = get_mean(config['delayICUD'])
    delay_icurec = get_mean(config['delayICUREC'])
    hosfrac =get_mean(config['hosfrac'])
    dfrac = get_mean(config['dfrac'])
    t_max = np.size(data[:, I_HOSCUM])
    time = np.linspace(1, t_max, int(t_max) )
    icufrac = read_icufrac_data(config, time,0)
    # icufrac =config['ICufrac']
    icudfrac = get_mean(config['icudfrac'])

    hos = data[:,I_HOS]

    if np.sum(np.cumsum(hos))>0.1:
        return data

    # else include the hospitalised column

    hoscum = data[:, I_HOSCUM]

    fracs = [hosfrac, dfrac, icufrac, icudfrac]
    delays = [delay_rec, delay_hos, delay_hosrec, delay_hosd, delay_icu, delay_icud, delay_icurec]

    dataprocess = do_hospitalization_process(hoscum, delays, fracs)
    hos = dataprocess[:,OP_HOS]
    hos = hos[:, None]
    icucum = dataprocess[:,OP_ICUCUM]
   # print ('day icufrac icucum')
   # for i, ic in enumerate(icucum):
   #     if (i<t_max):
   #         print (time[i], icufrac[i], ic)
    datanew = np.concatenate((data[:,0:6], hos), axis=-1)
    return datanew


def calc_axis_interval(num_days):
    if num_days < 70:
        interval = 7
    elif num_days<360:
        interval = 14
    else:
        interval = 28
    return interval
