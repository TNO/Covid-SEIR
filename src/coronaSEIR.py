import numpy as np
from scipy.integrate import odeint

from src.tools import do_hospitalization_process
from src.tools import gauss_smooth_shift

def base_seir_model(param, dict):
    """
    # S=susceptable, E=exposed, I=infected, R=Removed (with icfrac with a delay of idelayD)
    # Added: Recovered, Hospitalized, Dead
    :param param:
    :param dict:
    :return:
    """

    n, r0, sigma, gamma, alpha, delay_hos, delay_rec, delay_hosrec, delay_hosd, hosfrac, \
    dfrac, icufrac, delay_icu, delay_icud, delay_icurec, icudfrac, m, population, dayalpha, \
        rec_sd, hos_sd, hosrec_sd, hosd_sd, icu_sd, icud_sd, icurec_sd, icufracscale, ndataend  = parse_input(param, dict)
    t = dict['time']
    beta = r0 * gamma

    # Helper class to interpolate alpha to all times
    class AlphaFunc(object):
        def __init__(self, alpha, time, dayalpha):
            ifeltdelay = 1
            alpha_f = np.zeros_like(time)
            for j, dayr in enumerate(dayalpha):
                alpha_f[t.tolist().index(int(dayr)+ifeltdelay):] = alpha[j]

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
        #if (alpha > 0.98):
        #    print('alpha : ', alpha)
        return dydt

    time_inflation = 5  # To get smooth ODE response despite step functions in alpha
    t_new = np.linspace(t.min(), t.max(), (len(t) - 1) * time_inflation + 1)
    result = odeint(ode, init_vals, t_new,
                    args=(beta, sigma, gamma, alpha_t))

    # After ODE, select only timesteps of interest
    # Also scale to susceptible population
    result = result[::time_inflation, :] * population * m



    # Then add other metrics
    r = result[:, 3]
    #hoscum = np.roll(r, int(delay_hos)) * hosfrac
    #hoscum[:int(delay_hos)] = 0

    hoscum =  gauss_smooth_shift(r, delay_hos, hos_sd, scale=hosfrac)

    icufrac2 = icufrac
    if (not np.isscalar(icufrac)):
       icufrac2 = np.concatenate((icufrac[0:ndataend+3], icufrac[ndataend+3:]*icufracscale))

    fracs = [hosfrac, dfrac, icufrac2, icudfrac]
    delays= [delay_rec, delay_hos, delay_hosrec, delay_hosd, delay_icu, delay_icud, delay_icurec]
    gauss_sd= [rec_sd, hos_sd, hosrec_sd, hosd_sd, icu_sd, icud_sd, icurec_sd]
    gauss_sd = [0,0,0, 0, 0, icud_sd, icurec_sd]


    dataprocess = do_hospitalization_process(hoscum, delays, fracs, gauss_stddev=gauss_sd, removed=r)

    OP_HOS = 0
    OP_HOSCUM = 1
    OP_ICU = 2
    OP_ICUCUM = 3
    OP_REC = 4
    OP_DEAD = 5


    rec = dataprocess[:,OP_REC, None]
    hos = dataprocess[:,OP_HOS, None]
    icu = dataprocess[:,OP_ICU, None]
    icucum = dataprocess[:,OP_ICUCUM, None]
    hoscum2 = dataprocess[:,OP_HOSCUM, None]
    dead = dataprocess[:,OP_DEAD, None]

    cuminf = result[:,2] + result[:,3]



    t_out = (t - dict['time_delay'])[:,None]
    alpha_out = (1.0-alpha_t.get_alpha(t))[:,None]
    #alpha_out = (alpha_at_t - dict['time_delay'])[:, None]
    res_out = np.concatenate((t_out,result,hos,hoscum2,icu,icucum,rec,dead, cuminf[:,None], alpha_out ),axis=-1)
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

    rec_sd = get_from_dict('rec_sd', dict, param)
    hos_sd = get_from_dict('hos_sd', dict, param)
    hosrec_sd = get_from_dict('hosrec_sd', dict, param)
    hosd_sd = get_from_dict('hosd_sd', dict, param)
    icu_sd = get_from_dict('icu_sd', dict, param)
    icud_sd = get_from_dict('icud_sd', dict, param)
    icurec_sd = get_from_dict('icurec_sd', dict, param)
    icufracscale = get_from_dict('icufracscale', dict, param)
    ndataend = get_from_dict('ndataend', dict, param)


    return n, r0, sigma, gamma, alpha, delay_hos, delay_rec, delay_hosrec, delay_hosd, hosfrac, \
           dfrac,icufrac,delay_icu,delay_icud,delay_icurec,icudfrac, m, population, dayalpha, \
           rec_sd, hos_sd, hosrec_sd, hosd_sd, icu_sd, icud_sd, icurec_sd, icufracscale, ndataend
