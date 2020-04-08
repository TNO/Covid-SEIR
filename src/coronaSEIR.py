import numpy as np
from scipy.integrate import odeint


def base_seir_model(param, dict):
    """
    # S=susceptable, E=exposed, I=infected, R=Removed (with icfrac with a delay of idelayD)
    # Added: Recovered, Hospitalized, Dead
    :param param:
    :param dict:
    :return:
    """

    n, r0, sigma, gamma, alpha, delay_hos, delay_rec, delay_hosrec, delay_hosd, hosfrac, \
    dfrac, icufrac, delay_icu, delay_icud, delay_icurec, icudfrac, m, population, dayalpha = parse_input(param, dict)
    t = dict['time']
    beta = r0 * gamma

    # Helper class to interpolate alpha to all times
    class AlphaFunc(object):
        def __init__(self, alpha, time, dayalpha):
            ifeltdelay = 1
            alpha_f = np.zeros_like(time)
            for j, dayr in enumerate(dayalpha):
                alpha_f[t.tolist().index(dayr+ifeltdelay):] = alpha[j]

            self.alpha = alpha_f
            self.time = time

        def get_alpha(self, tx):
            return np.interp(tx, self.time, self.alpha)

    init_vals = 1. - (1. / n), 1. / n, 0, 0

    alpha_t = AlphaFunc(alpha, t, dayalpha)

    def ode(y, t, beta, sigma, gamma, alpha_t):
        s, e, i, r = y

        # Basic SEIR with time-dependent alpha
        alpha = alpha_t.get_alpha(t)
        dsdt = (alpha - 1) * beta * s * i
        dedt = (1 - alpha) * beta * s * i - sigma * e
        didt = sigma * e - gamma * i
        drdt = gamma * i

        dydt = [dsdt,  # dS/dt      Susceptible
                dedt,  # dE/dt      Exposed
                didt,  # dI/dt      Infected
                drdt  # dR/dt      Removed
                ]

        return dydt

    time_inflation = 5  # To get smooth ODE response dispite step functions in alpha
    t_new = np.linspace(t.min(), t.max(), (len(t) - 1) * time_inflation + 1)
    result = odeint(ode, init_vals, t_new,
                    args=(beta, sigma, gamma, alpha_t))

    # After ODE, select only timesteps of interest
    # Also scale to susceptible population
    result = result[::time_inflation, :] * population * m



    # Then add other metrics
    r = result[:, 3]
    hoscum = np.roll(r, int(delay_hos)) * hosfrac
    hoscum[:int(delay_hos)] = 0


    #icufrac is the fraction of patients going to ICU at the time of entry in hospital
    icufrac2 = icufrac
    # dp frac is the fraction of all  patients dying who did not go to ICU, referenced at entry in the hospital
    dpfrac = (dfrac - icufrac * icudfrac)/(1.0001-icufrac)  #/ (1 - icufrac * icudfrac)
    hosday2 = np.concatenate((np.array([0]), np.diff(hoscum)))
    hosday = hosday2*1.0
    if (delay_icu>0.01):
        for i, num in enumerate(hosday):
            irange = int(delay_icu*2)
            ilow =int (max(0,(i-irange)))
            ihigh = int(min(np.size(hosday),i+1 ))
            hosday[i] = np.mean( hosday2[ilow:ihigh])
   # check for consistency
    hoscum2 = np.cumsum(hosday)
    # construct hospitalization and icu as if nobody dies in the hospital, except icu
    icuday = hosday*icufrac
    icucum = np.cumsum(icuday)
   # icucum = np.roll(icucum, int(delay_icu))
   # icucum[:int(delay_icu)] = 0

    #recovered from icu, taking into account deaths from icu
    icu_rechos = np.roll(icucum, int(delay_icurec))  * (1. - icudfrac) # Back to hospital
    icu_rechos[:int(delay_icurec)] = 0
    icu_recfull = np.roll(icu_rechos, int(delay_hosrec)) # All who go back to hospital recover, but this takes time
    # this is the full recovered patients who went through ICU
    icu_recfull[:int(delay_hosrec)] = 0

    # dead from icu
    icu_dead = np.roll(icucum, int(delay_icud)) * icudfrac
    icu_dead[: int(delay_icud)] = 0

    # this is the full rovered patients who did not went through ICU
    rechos = np.roll(hoscum *(1-icufrac)*(1. - dpfrac), int(delay_hosrec))
    rechos[:int(delay_hosrec)] = 0
    # dead from hospitalized but not including icu
    hdead = np.roll(hoscum*(1-icufrac)* dpfrac, int(delay_hosd))
    hdead[:int(delay_hosd)] = 0

    recmild = np.roll(r, int(delay_rec)) * (1. - hosfrac)
    recmild[:int(delay_rec)] = 0

    # actual icu is cumulative icu minus icu dead and icu revovered
    icu = icucum - icu_dead - icu_rechos
    dead = hdead + icu_dead
    hos = hoscum - rechos - icu_recfull - dead # includes ICU count
    rec = rechos + recmild + icu_recfull

    rec = rec[:, None]
    hos = hos[:, None]
    icu = icu[:, None]
    icucum = icucum[:,None]
    hoscum = hoscum[:, None]
    dead = dead[:, None]

    # Check balances
    # sick_home = r - hos - rec - icu - dead
    # check = sick_home + dead + rec + icu + hos
    # np.allclose(r, check)

    t_out = (t - dict['time_delay'])[:,None]
    res_out = np.concatenate((t_out,result,hos,hoscum,icu,icucum,rec,dead),axis=-1)
    # Time, Suscep, Expos, Infec, Removed, Hospitalized, Hos (cummulative), ICU, ICU (cummulative), Recovered, Dead
    return res_out.T


def parse_input(param, dict):
    def get_from_dict(name, dict, param):
        try:
            ret = dict['locked'][name]
        except KeyError:
            param_ind = np.array(dict['free_param']) == name
            ret = np.array(param)[param_ind]
            if len(ret) == 1:
                ret = ret[0]
        return ret

    n = get_from_dict('n', dict, param)
    r0 = get_from_dict('r0', dict, param)
    sigma = get_from_dict('sigma', dict, param)
    gamma = get_from_dict('gamma', dict, param)
    alpha = get_from_dict('alpha', dict, param)
    delay_hos = get_from_dict('delay_hos', dict, param)
    delay_rec = get_from_dict('delay_rec', dict, param)
    delay_hosrec = get_from_dict('delay_hosrec', dict, param)
    delay_hosd = get_from_dict('delay_hosd', dict, param)
    hosfrac = get_from_dict('hosfrac', dict, param)
    dfrac = get_from_dict('dfrac', dict, param)
    icufrac = get_from_dict('icufrac', dict, param)
    delay_icu = get_from_dict('delay_icu', dict, param)
    delay_icud = get_from_dict('delay_icud', dict, param)
    delay_icurec = get_from_dict('delay_icurec', dict, param)
    icudfrac = get_from_dict('icudfrac', dict, param)
    m = get_from_dict('m', dict, param)
    population = get_from_dict('population', dict, param)
    dayalpha = get_from_dict('dayalpha', dict, param)

    return n, r0, sigma, gamma, alpha, delay_hos, delay_rec, delay_hosrec, delay_hosd, hosfrac, \
           dfrac,icufrac,delay_icu,delay_icud,delay_icurec,icudfrac, m, population, dayalpha
