import sys
from src.io_func import load_config, load_data, hospital_forecast_to_txt
from src.parse import parse_config, reshape_prior
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from vis.general import plot_posterior
from tqdm import tqdm
from src.tools import generate_hos_actual

from src import coronaSEIR



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


def run_prior(config):
    """
    Create a prior ensemble of model runs.
    :param config: The entire model configuration in dict-form
    :return: Forward model results, the parameter sets used to create them, and an array of timesteps
    """
    np.random.seed(1)
    # Parse parameters
    m_prior, fwd_args = parse_config(config)
    m_prior = reshape_prior(m_prior)

    # Run simulation
    results = np.zeros((len(m_prior), 11, len(fwd_args['time'])))
    for i in tqdm(range(len(m_prior)), desc='Running prior ensemble'):
        param = m_prior[i]
        results[i] = coronaSEIR.base_seir_model(param,fwd_args)

    tnew = fwd_args['time'] - fwd_args['time_delay']

    return results, tnew, m_prior, fwd_args


def calibrate_with_data(prior, t, data,data_index, output_index):
    """
    Use the observed cases ('hospitalized' or 'dead') to assign each prior member a probability
    :param prior: The prior ensemble
    :param t: The time definition
    :param data: The observed data
    :param model_conf: The model configuration for each prior member
    :param mode: Calibration mode ('hospitalized' or 'dead')
    :return: Probability of each prior member (sums to 1.0)
    """

    # Find the timesteps in the model corresponding to the data
    t_ind = [np.where(t == a)[0][0] for a in data[:, I_TIME]]
    forecasted_cases = prior[:, output_index, t_ind]
    observed_cases = data[:, data_index]
    total_cases_forecasted = forecasted_cases[:, -1] - forecasted_cases[:,0];
    forecasted_rate = np.diff(forecasted_cases, axis=-1)
    forecasted_rate = np.clip(forecasted_rate, a_min=1e-05, a_max=None)
    observed_rate = np.diff(observed_cases)

    ll = np.sum(observed_rate[None, :] * np.log(forecasted_rate), axis=-1) - total_cases_forecasted
    ll = ll.astype(float)
    llmax = ll.max()

    p = np.exp(ll - llmax) / np.sum(np.exp(ll - llmax))

    return p


def integrate_calibration(results, p):
    # Multiply each prior member with its probability. Sum to obtain mean posterior
    integrated_result = np.einsum('ijk,i', results, p)
    return integrated_result


def rerun_model_with_alpha_uncertainty(model_conf, p, config,fwd_args):
    """
    Sample from the posterior distribution, and add extra samples to sufficiently account for (prior) uncertainty in
    alpha
    :param model_conf: Model configuration of the prior members
    :param p: Probality of the prior members
    :param config: Dict with model configuration
    :return: Forward models, sampled from posterior, with added alpha uncertainty
    """
    model_conf = np.array(model_conf)
    nr_samples = config['nr_forecast_samples']
    sample_indices = np.random.choice(np.arange(len(p)), size=nr_samples, p=p)

    # Sample from posterior
    post = model_conf[sample_indices]
    # Replace alpha uncertainty with prior uncertainty
    alpha_sample_indices = np.random.choice(np.arange(len(p)), size=nr_samples)
    for i, param in enumerate(fwd_args['free_param']):
        if param =='alpha':
            post[:,i] = model_conf[alpha_sample_indices,i]

    # Run simulation
    results = np.zeros((post.shape[0], 11, len(fwd_args['time'])))
    for i in tqdm(range(post.shape[0]), desc='Running posterior ensemble'):
        param = post[i]
        results[i] = coronaSEIR.base_seir_model(param, fwd_args)
    return results





def plot_results(results, t, configpath, config, data, integrated_result=None, cal_mode=None, prior=True, plot_plume= True):
    base = (os.path.split(configpath)[-1]).split('.')[0]
    outpath = os.path.join(os.path.split(os.getcwd())[0], 'output', base)

    fig_base = configpath.split('/')[-1][0:-5]
    fig, ax = plt.subplots()
    nr_samples = results.shape[0]
    transparancy = max(min(5.0 / nr_samples, 1),0.01)

    alphaday = config["dayalpha"]
    for i in alphaday:
        ax.axvline(x=i, color='silver')

    if (plot_plume):
        ax.plot(t, np.full_like(t, fill_value=np.nan), color='magenta', label='exposed')
        ax.plot(t, np.full_like(t, fill_value=np.nan), color='red', label='infected')
        ax.plot(t, np.full_like(t, fill_value=np.nan), color='green', label='hospitalized')
        ax.plot(t, np.full_like(t, fill_value=np.nan), color='blue', label='deceased')
        ax.plot(t, np.full_like(t, fill_value=np.nan), color='black', label='recovered')

        ax.plot(t, results[0:, O_EXP, :].T, color='magenta', alpha=transparancy)
        ax.plot(t, results[0:, O_INF, :].T + results[0:, O_REM, :].T, color='red', alpha=transparancy)
        ax.plot(t, results[0:, O_HOSCUM, :].T, color='green', alpha=transparancy)
        ax.plot(t, results[0:, O_DEAD, :].T, color='blue', alpha=transparancy)
        ax.plot(t, results[0:, O_REC, :].T, color='black', alpha=transparancy)


    plt.xlabel('Time [days]')
    plt.ylabel('Number')

    y_lim = config['YMAX']
    x_lim = config['XMAX']
    plt.ylim(1, y_lim)
    plt.xlim(1, x_lim)

    if config['plot']['y_axis_log']:
        plt.semilogy()

    plt.grid(True)
    ax.legend(loc=config['plot']['legendloc'], fontsize=config['plot']['legendfont'])
    plt.savefig('{}_priors_MC_{}.png'.format(outpath, cal_mode), dpi=300)

    if not config['worldfile']:
        plt.plot(data[:, 0], data[:, I_HOSCUM], "go", markersize=2, markeredgecolor='k', alpha=1, label='hospitalized cum (data)')
    plt.plot(data[:, 0], data[:, I_INF], "r<", markersize=2, markeredgecolor='k', alpha=1, label='infected cum (data)')
    plt.plot(data[:, 0], data[:, I_DEAD], "bx", markersize=2, markeredgecolor='k', alpha=1, label='deceased(data)')
    plt.plot(data[:, 0], data[:, I_REC], ">", markersize=2, markeredgecolor='k', alpha=1, label='recovered(data)')


    if type(integrated_result) == np.ndarray:
        cuminfected  = integrated_result[O_INF] +integrated_result[O_REM]
        if plot_plume:
            if not config['worldfile']:
                ax.plot(t, integrated_result[O_HOSCUM], color='green')
            ax.plot(t, cuminfected, color='red')
            ax.plot(t, integrated_result[O_DEAD], color='blue')
            ax.plot(t, integrated_result[O_EXP], color='magenta')
            ax.plot(t, integrated_result[O_REC], color='black')
        else:
            ax.plot(t, integrated_result[O_EXP], color='magenta', label='exposed')
            ax.plot(t, cuminfected, color='red', label='infected')
            if not config['worldfile']:
                ax.plot(t, integrated_result[O_HOSCUM], color='green', label='hospitalized')

            ax.plot(t, integrated_result[O_DEAD], color='blue',label='deceased' )
            ax.plot(t, integrated_result[O_REC], color='black',label='recovered' )



    ax.legend(loc=config['plot']['legendloc'], fontsize=config['plot']['legendfont'])

    plt.title('Calibrated on {}'.format(cal_mode))
    plottype = '{}_forecast_MC_{}'
    if prior:
        plottype = '{}_hindcast_MC_{}.png'
    plt.savefig(plottype.format(outpath, cal_mode), dpi=300)
    plt.close()



def sample_from_prior(prior,p,config):
    nr_samples = config['nr_forecast_samples']
    sample_indices = np.random.choice(np.arange(len(p)), size=nr_samples, p=p)
    return prior[sample_indices]

def main(configpath):
    # Load the model configuration file and the data (observed cases)
    config = load_config(configpath)
    data = load_data(config)



    useworldfile = config['worldfile']
    if (not useworldfile):
        data = generate_hos_actual(data, config)
    # Run the forward model to obtain a prior ensemble of models
    prior, time, prior_param, fwd_args = run_prior(config)

    # Calibrate the model (as calibration data, we either use 'hospitalized' or 'dead')
    if config['calibration_mode'] == 'dead':
        data_index = I_DEAD
        output_index = O_DEAD
    elif config['calibration_mode'] == 'hospitalized':
        data_index = I_HOS
        output_index = O_HOS
    elif config['calibration_mode'] == 'hospitalizedcum':
        data_index = I_HOSCUM
        output_index = O_HOSCUM
    elif config['calibration_mode'] == 'ICU':
        data_index = I_ICU
        output_index = O_ICU
    else:
        raise NotImplementedError

    plotplume = config['plot']['hindcast_plume']

    p = calibrate_with_data(prior, time, data, data_index, output_index)  # Posterior probability for each ensemble member
    integrated_results = integrate_calibration(prior, p)  # Mean posterior model

    # Create output of the calibration phase
    plot_results(prior, time, configpath, config, data, integrated_results, config['calibration_mode'], prior=True, plot_plume= plotplume )

    # Based on the model performance in the past, run models again, forecasting the future 

    forecast = rerun_model_with_alpha_uncertainty(prior_param, p, config, fwd_args)
    base = (os.path.split(configpath)[-1]).split('.')[0]
    outbase = os.path.join(os.path.split(os.getcwd())[0], 'output', base)

    plot_results(forecast, time, configpath, config, data, cal_mode= config['calibration_mode'], prior=False,
                     plot_plume=True)

    plot_posterior(forecast, config,outbase,data,['time'])

    try:
        hospital_forecast_to_txt(forecast, time, configpath, config, data)
    except:
        pass


if __name__ == '__main__':
    # This script only accepts one input argument: the path to the configuration .json file
    main(sys.argv[1])
