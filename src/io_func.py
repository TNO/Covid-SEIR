import json
import os
import h5py
import numpy as np
import res.global_data as global_data
# import res.world_data as world_data


def load_config(configpath):
    """
    Load the configuration file
    :param configpath: Path to configuration .json file
    :return: dict with the configuration
    """
    with open(configpath, 'r') as f:
        config = json.load(f)
    try:
        assert np.all(len(config['alpha']) == len(config['dayalpha'])), 'Not the same alphas as alphadays'
        assert np.all(config['dayalpha'] == sorted(config['dayalpha'])), 'Dayalpha has to be chronological'
    except KeyError:
        pass
    return config

def save_input_data(configpath,  datanew):
    base = (os.path.split(configpath)[-1]).split('.')[0]
    outpath = os.path.join(os.path.split(os.getcwd())[0], 'output', base)
    header = 'day,test,dead,rec,hoscum,icu,hosact'
    table =  datanew[:,0:7] #np.concatenate((datanew[:, 0:6]), axis=-1)
    np.savetxt('{}_inputdata{}.txt'.format(outpath, '', ''),
               table, header=header, delimiter=',',comments='',fmt='%8d')



def load_data(config):
    """
    Load the observed data, either from a world-wide datafile, or from a country-specific on
    # data format
    #  column 0 -  day
    #  column 1 - cumulative registered infected
    #  column 2  - cum  dead
    #  column 3 - cum recovered
    #  --- extension for own recorded data (not from world file)
    #  column 4 - cumulative hospitalized
    #  column 5 - actual IC units used
    :param config: dict with configuration
    :return: the observed data in an array
    """

    direc = '.'
    useworldfile = config['worldfile']
    country = config["country"]
    province = ''
    startdate = ''
    firstdate = ''
    if 'province' in config:
        province = config['province']
    data_offset = 0
    if 'startdate' in config:
        startdate = config['startdate']
    if useworldfile:
        # wdata, firstdate = world_data.get_country_xcdr(country, 'all', dateOffset=data_offset)
        # print(firstdate)
        # data = np.array(wdata)
        data, firstdate = global_data.get_country_data(country, province, startdate)
        print(firstdate)
    else:
        data = np.loadtxt(os.path.join(direc, country))
    if 'maxrecords' in config:
        n = config['maxrecords']
        data = data[:n]
    return data,firstdate


def  read_icufrac_data(config, time, time_delay):
    icufracfile = 'None'
    icufrac= 'None'
    if 'icufracfile' in config:
        icufracfile = config['icufracfile']
        direc = '.'
        # read icufracs from a datafile, these are
        data =np.genfromtxt(os.path.join(direc, icufracfile), names=True)
        #data = np.loadtxt(os.path.join(direc, icufracfile), names=True, delimiter=' ')
        icudata = data['icufrac']
        if 'maxrecords' in config:
            n = config['maxrecords']
            icudata = icudata[:n]
        # map the data to array equal to length of time
        icufrac = np.ones_like(time)
        for i, num in enumerate(icufrac):
            if (i<np.size(icudata)):
                icufrac[i] = icudata[i]
            else:
                icufrac[i] = icufrac[i-1]
         # delay the icufrac
        icufrac = np.roll(icufrac, int(time_delay))
        # set the first times the icufrac to the first value
        icufrac[:int(time_delay)] = icudata[0]
    else:
        icufrac = config['ICufrac']

    return icufrac



def save_results(results, fwd_args, config, outpath, data, mode):
    # Remove old output
    if os.path.exists(outpath):
        os.remove(outpath)

    with h5py.File(outpath, 'w') as hf:
        # Config
        hf['mode'] = mode
        conf = hf.create_group('config')
        dict_to_h5(conf, config)

        # Data
        datag = hf.create_group('data')
        datag['data'] = data
        string_array_to_h5(datag, 'labels',['time', 'infected', 'dead', 'recovered', 'hospitalizedcum', 'icu','hospitalized'])

        if mode == 'esmda':
            # Parameters
            param = hf.create_group('parameters')
            pp_array = np.array(results['M'][0])
            pos_array = np.array(results['M'][-1])
            labelled_array_to_h5(param, 'prior', pp_array, ['members', 'free_param'],
                                 [np.arange(pp_array.shape[0]), fwd_args['free_param']])
            labelled_array_to_h5(param, 'posterior', pos_array, ['members', 'free_param'],
                                 [np.arange(pos_array.shape[0]), fwd_args['free_param']])

            # Forward model
            model = hf.create_group('model')
            prior = np.array(results['fw'][0])
            posterior = np.array(results['fw'][-1])
            # Time, Suscep, Expos, Infec, Removed, Hospitalized, Hos (cummulative), ICU, ICU (cummulative), Recovered, Dead
            names = ['time', 'susceptible', 'exposed', 'infected', 'removed', 'hospitalized', 'hospitalized_cum', 'icu',
                     'icu_cum', 'recovered', 'dead']
            labelled_array_to_h5(model, 'prior', prior, ['members', 'forecasted_numbers'],
                                 [np.arange(prior.shape[0]), names])
            labelled_array_to_h5(model, 'posterior', posterior, ['members', 'forecasted_numbers'],
                                 [np.arange(posterior.shape[0]), names])

        elif mode == 'mcmc':
            pass

        else:
            raise NotImplementedError


def dict_to_h5(base, dictionary):
    keys = dictionary.keys()
    for key in keys:
        if type(dictionary[key]) != dict:
            base[key] = dictionary[key]
        else:
            new_base = base.create_group(key)
            dict_to_h5(new_base, dictionary[key])

def dict_from_h5(hf):
    ret_dict = {}
    keys = hf.keys()
    for key in keys:
        try:
            ret_dict[key] = hf[key][()]
        except AttributeError:
            ret_dict[key] = dict_from_h5(hf[key])
    return ret_dict


def hospital_forecast_to_txt(results, t, configpath, config, data):
    hospitalized = results[:, 6, :].T
    steps = np.arange(1., 83.)
    t_ind = [np.where(t == a)[0][0] for a in steps]
    hospitalized = hospitalized[t_ind]
    p_values = config['p_values']
    p_array = []
    posterior_length = hospitalized.shape[1]
    for hosp_day in hospitalized:
        array_sorted = np.sort(hosp_day)
        p_array.append([array_sorted[int(posterior_length * p)] for p in p_values])
    mean = np.mean(hospitalized, axis=1)
    h_pvalues = ['P' + str(int(100 * a)) for a in p_values]
    header = 'time,mean,' + ','.join(h_pvalues) + ',observed'
    p_array = np.asarray(p_array)
    observed = np.pad(data[:, 4], (0, len(steps) - data.shape[0]), mode='constant', constant_values=np.nan)[:, None]
    table = np.concatenate((steps[:, None], mean[:, None], p_array, observed), axis=1)
    np.savetxt('output/' + configpath.split('.')[0] + '_hospitalized_prob.csv', table, header=header, delimiter=',',
               comments='', fmt='%.2f')


def labelled_array_to_h5(base, name, data, labels, scales):
    base[name] = data
    for i, _ in enumerate(labels):
        base[name].dims[i].label = labels[i]
        try:
            try:
                base[labels[i]] = scales[i]
            except TypeError:
                string_array_to_h5(base, labels[i], scales[i])
                base[labels[i]].make_scale()
            base[name].dims[i].attach_scale(base[labels[i]])
        except Exception as e:
            if 'Unable to create link (name already exists)' in str(e):
                base[name].dims[i].attach_scale(base[labels[i]])
            else:
                raise e


def string_array_to_h5(base, name, array):
    base[name] = [a.encode('ascii', 'ignore') for a in array]


def string_array_from_h5(array):
    return [str(a).strip('b').strip(r"\'") for a in array]


def labelled_array_from_h5(address):
    array = address[()]
    labels = [dim.label for dim in address.dims]
    scales = [np.array(dim.values())[0] for dim in address.dims]
    for i, scale in enumerate(scales):
        if scale.dtype != float and scale.dtype != int:
            scales[i] = string_array_from_h5(scale)

    return array, labels, scales
