import numpy as np
from scipy.stats import uniform, norm
from scipy.stats._distn_infrastructure import rv_frozen

from src.io_func import read_icufrac_data


def parse_config(config, mode='prior'):
    """
    Turns configuration file into parsed model parameters
    :param config: The entire configuration dictionary
    :return: parsed model parameters: params, init_values, alphas_t, m, time, time_delay
    """
    nr_samples = config['nr_prior_samples'] if mode == 'prior' else config['nr_forecast_samples']
    t_max = config['t_max']
    time_delay = config['time_delay']
    population = to_distr(config['population'], nr_samples)
    n = to_distr(config['N'], nr_samples)
    sigma = to_distr(config['sigma'], nr_samples)
    gamma = to_distr(config['gamma'], nr_samples)
    r0 = to_distr(config['R0'], nr_samples)
    alpha = np.array(config['alpha'])
    dayalpha1 = config['dayalpha']
    dayalpha = np.array(dayalpha1)
    dayalpha = (dayalpha + time_delay)
    delay_rec = to_distr(config['delayREC'], nr_samples)
    delay_hos = to_distr(config['delayHOS'], nr_samples)
    delay_hosrec = to_distr(config['delayHOSREC'], nr_samples)
    delay_hosd = to_distr(config['delayHOSD'], nr_samples)
    delay_icu = to_distr(config['delayICUCAND'], nr_samples)
    delay_icud = to_distr(config['delayICUD'], nr_samples)
    delay_icurec = to_distr(config['delayICUREC'], nr_samples)
    hosfrac = to_distr(config['hosfrac'], nr_samples)
    dfrac = to_distr(config['dfrac'], nr_samples)

    # add standard dev for gaussian smoothing
    rec_sd = get_gauss_smooth_sd(config['delayREC']);
    hos_sd = get_gauss_smooth_sd(config['delayHOS']);
    hosrec_sd = get_gauss_smooth_sd(config['delayHOSREC']);
    hosd_sd = get_gauss_smooth_sd(config['delayHOSD']);

    icu_sd = get_gauss_smooth_sd(config['delayICUCAND']);
    icud_sd = get_gauss_smooth_sd(config['delayICUD']);
    icurec_sd = get_gauss_smooth_sd(config['delayICUREC']);



    icudfrac = to_distr(config['icudfrac'], nr_samples)
    m = to_distr(config['m'], nr_samples)
    time = np.linspace(0, t_max, int(t_max) + 1)
    icufrac = read_icufrac_data(config, time, time_delay)
    # indep_scaling = [to_distr(config['alphascaling'], nr_samples) for _ in alpha]
    # Make scaling fully correlated
    # alpha_n = [a * indep_scaling[0] for i, a in enumerate(alpha)]
    # Make scalling fully uncorrelated
    # alpha_n = [a * indep_scaling[i] for i, a in enumerate(alpha)]
    #
    # alpha_out = np.array(alpha_n).T.clip(max=0.99)
    # alpha_out = [list(a) for a in alpha_n]
    alpha_dict = [{'type': 'uniform', 'min': a[0], 'max': a[1]} for a in alpha]
    alpha_n = [to_distr(a, nr_samples) for a in alpha_dict]

    return_dict = {'free_param': [], 'm_prior': [], 'locked': {}}
    if np.size(icufrac) == np.size(time):
        return_dict['locked']['icufrac'] = icufrac
        params = [n, r0, sigma, gamma, alpha_n, delay_hos, delay_rec, delay_hosrec, delay_hosd, delay_icu, delay_icud,
                  delay_icurec, hosfrac, dfrac, icudfrac, m, population,
                  icurec_sd, icud_sd, icu_sd, rec_sd, hos_sd, hosrec_sd, hosd_sd]
        names = ['n', 'r0', 'sigma', 'gamma', 'alpha', 'delay_hos', 'delay_rec', 'delay_hosrec', 'delay_hosd',
                 'delay_icu', 'delay_icud', 'delay_icurec', 'hosfrac', 'dfrac', 'icudfrac', 'm',
                 'population',
                 'icured_sd', 'icud_sd','icu_sd','rec_sd', 'hos_sd', 'hosrec_sd', 'hosd_sd']
    else:
        params = [n, r0, sigma, gamma, alpha_n, delay_hos, delay_rec, delay_hosrec, delay_hosd, delay_icu, delay_icud,
                  delay_icurec, hosfrac,icufrac,dfrac,icudfrac, m, population,
                  icurec_sd, icud_sd, icu_sd, rec_sd, hos_sd, hosrec_sd, hosd_sd
                  ]
        names = ['n', 'r0', 'sigma', 'gamma', 'alpha', 'delay_hos', 'delay_rec', 'delay_hosrec', 'delay_hosd',
                 'delay_icu','delay_icud','delay_icurec','hosfrac','icufrac', 'dfrac','icudfrac', 'm', 'population',
                 'icured_sd', 'icud_sd','icu_sd','rec_sd', 'hos_sd', 'hosrec_sd', 'hosd_sd'
                    ]

    return_dict['locked']['dayalpha'] = dayalpha
    return_dict['locked']['icu_sd'] = icu_sd
    return_dict['locked']['icud_sd'] = icud_sd
    return_dict['locked']['icurec_sd'] = icurec_sd
    return_dict['locked']['rec_sd'] = rec_sd
    return_dict['locked']['hos_sd'] = hos_sd
    return_dict['locked']['hosrec_sd'] = hosrec_sd
    return_dict['locked']['hosd_sd'] = hosd_sd

    for i, param in enumerate(params):
        name = names[i]
        return_dict = add_to_dict(return_dict, param, name)
    return_dict['time'] = time
    return_dict['time_delay'] = time_delay

    return return_dict['m_prior'], return_dict

def parse_config_mcmc(config):
    """
    Turns configuration file into parsed model parameters
    :param config: The entire configuration dictionary
    :return: parsed model parameters: params, init_values, alphas_t, m, time, time_delay
    """
    t_max = config['t_max']
    time_delay = config['time_delay']
    population = to_distr(config['population'], 'mcmc')
    n = to_distr(config['N'], 'mcmc')
    sigma = to_distr(config['sigma'], 'mcmc')
    gamma = to_distr(config['gamma'], 'mcmc')
    r0 = to_distr(config['R0'], 'mcmc')
    alpha = np.array(config['alpha'])
    dayalpha1 = config['dayalpha']
    dayalpha = np.array(dayalpha1)
    dayalpha = (dayalpha + time_delay)
    delay_rec = to_distr(config['delayREC'], 'mcmc')
    delay_hos = to_distr(config['delayHOS'], 'mcmc')
    delay_hosrec = to_distr(config['delayHOSREC'], 'mcmc')
    delay_hosd = to_distr(config['delayHOSD'], 'mcmc')
    delay_icu = to_distr(config['delayICU'], 'mcmc')
    delay_icud = to_distr(config['delayICUD'], 'mcmc')
    delay_icurec = to_distr(config['delayICUREC'], 'mcmc')
    hosfrac = to_distr(config['hosfrac'], 'mcmc')
    dfrac = to_distr(config['dfrac'], 'mcmc')
    icufrac = to_distr(config['ICufrac'], 'mcmc')
    icudfrac = to_distr(config['icudfrac'], 'mcmc')
    m = to_distr(config['m'], 'mcmc')
    time = np.linspace(0, t_max, int(t_max) + 1)
    # indep_scaling = [to_distr(config['alphascaling'], 'mcmc') for _ in alpha]
    alpha_dict = [{'type': 'uniform', 'min': a[0], 'max': a[1]} for a in alpha]
    alpha_n = [to_distr(a,'mcmc') for a in alpha_dict]

    params = [n, r0, sigma, gamma, alpha_n, delay_hos, delay_rec, delay_hosrec, delay_hosd, delay_icu, delay_icud,
              delay_icurec, hosfrac,icufrac,dfrac,icudfrac, m, population]
    names = ['n', 'r0', 'sigma', 'gamma', 'alpha', 'delay_hos', 'delay_rec', 'delay_hosrec', 'delay_hosd',
             'delay_icu','delay_icud','delay_icurec','hosfrac','icufrac', 'dfrac','icudfrac', 'm', 'population']
    return_dict = {'free_param': [], 'm_prior': [], 'locked': {}}
    return_dict['locked']['dayalpha'] = dayalpha

    for i, param in enumerate(params):
        name = names[i]
        return_dict = add_to_dict_mcmc(return_dict, param, name)
    return_dict['time'] = time
    return_dict['time_delay'] = time_delay

    return return_dict

def add_to_dict_mcmc(dict, dist, name):
    def add_rv(dict,dist,name):
        if dist.kwds['scale'] == 1e-9:
            dict['locked'][name] = dist.kwds['loc'] + 5e-10
        else:
            dict['m_prior'].append(dist)
            dict['free_param'].append(name)
        return dict

    if type(dist) == rv_frozen:
        add_rv(dict, dist, name)
    else:
        if type(dist[0])!=rv_frozen:
            dict['locked'][name] = dist
        else:
            for rv in dist:
                dict = add_rv(dict,rv,name)

    return dict



def add_to_dict(dict, array, name):
    array = np.array(array).T
    if array.max()-array.min()<=5e-9:
        dict['locked'][name] = array.mean()
    else:
        if len(array.shape) == 1:
            dict['m_prior'].append(array)
            dict['free_param'].append(name)
        elif len(array.shape) > 1:
            for i in range(array.shape[1]):
                dict['m_prior'].append(array[:, i])
                dict['free_param'].append(name)
    return dict



def get_gauss_smooth_sd(var):
    """
    Sample from distributions
    :param var: either a dict describing a distribution, or a scalar value
    :return: mean value
    """
    if type(var) == dict:
        if var['type'] == 'uniform':
            try:
                rv = var['smooth_sd']
            except:
                 rv = 0
        elif var['type'] == 'normal':
            try:
                rv = var['smooth_sd']
            except:
                 rv = 0
        else:
            raise NotImplementedError('Distribution not implemented')
    else:
        rv = 0


    ret = rv

    return ret


def get_mean(var):
    """
    Sample from distributions
    :param var: either a dict describing a distribution, or a scalar value
    :return: mean value
    """
    if type(var) == dict:
        if var['type'] == 'uniform':
            rv = 0.5 * (var['max'] + var['min'])
        elif var['type'] == 'normal':
            rv = loc=var['mean']
        else:
            raise NotImplementedError('Distribution not implemented')
    else:
        rv = var


    ret = rv

    return ret

def to_distr(var, nr_samples):
    """
    Sample from distributions
    :param var: either a dict describing a distribution, or a scalar value
    :param nr_samples: number of samples required
    :return: 1D array of sampled values
    """
    if type(var) == dict:
        if var['type'] == 'uniform':
            rv = uniform(loc=var['min'], scale=var['max'] - var['min'])
        elif var['type'] == 'normal':
            rv = norm(loc=var['mean'], scale=var['stddev'])
        else:
            raise NotImplementedError('Distribution not implemented')
    else:
        rv = uniform(loc=var-5e-10, scale=1e-9)

    if nr_samples == 'mcmc':
        ret = rv
    else:
        ret = rv.rvs(nr_samples).clip(min=0)

    return ret


def reshape_prior(params):
    return [[entry[i] for entry in params] for i in range(len(params[0]))]
