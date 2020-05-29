import os
import warnings

from src.io_func import load_config, load_data, read_icufrac_data
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.special as sps


from src.parse import get_mean, to_gauss_smooth_sd
from src.tools import gauss_smooth_shift, do_hospitalization_process, calc_axis_interval


def find_lognormal_musd (mu_target, sd_target):
    mu = 0
    sd = 0
    mu_log = 0
    sd_log = 0.7


    while(abs(sd_target-sd)>1e-1):
        while (abs(mu_target -mu)>1e-2 ):
            mu = np.exp(mu_log + sd_log ** 2 / 2)
            mu_log = mu_log + np.log(mu_target/mu)
            sd = np.sqrt(np.exp(sd_log ** 2 + 2 * mu_log) * (np.exp(sd_log ** 2) - 1))
            print ('mu, sd ', mu,sd)
        sd_log = sd_log + 0.3* np.log(sd_target /sd)
        mu = np.exp(mu_log + sd_log ** 2 / 2)
    return mu_log, sd_log



def pdf_lognormal (x, mu_log, sd_log):
    r = (1.0/(x*sd_log * np.sqrt(2*np.pi))) *  np.exp(- (np.log(x)- mu_log)**2 / (2*sd_log**2) )
    return r

def find_gamma_ktheta (mu_target, sd_target):
    if (sd_target>mu_target):
        sd_target = mu_target-1
    theta = sd_target ** 2 / mu_target
    k = mu_target / theta
    return k,theta

def pdf_gamma(x, shape, scale):
        return (x ** (shape - 1) * (np.exp(-x / scale) / (sps.gamma(shape) * scale ** shape)))

def plot ():

        fig, ax = plt.subplots()
        plt.figure(figsize=(6.0, 6.0))


        plt.legend(loc='upper left')
        plt.xlabel('x-value',fontsize=12)
        plt.ylabel('pdf',fontsize=12)

        mu_target = 24
        sd_target = 19

        mu_log, sd_log = find_lognormal_musd(mu_target, sd_target)
        #mu_log = 2
        #sd_log = 0.7

        mu = np.exp(mu_log + sd_log**2/2)
        sd = np.sqrt(np.exp(sd_log**2+2*mu_log) * (np.exp(sd_log**2)-1))

        print ('mu, sd ', mu, sd )

        x_val = np.arange(0.1, 100, 0.2)
        y_val = pdf_lognormal (x_val, mu_log, sd_log)
        plt.plot(x_val, y_val, c='blue', lw=2)
        #x_val = np.arange(0.1, 50, 0.2)

        k,theta = find_gamma_ktheta(mu_target, sd_target)
        y_val2 = pdf_gamma(x_val, k,theta)
        plt.plot(x_val, y_val2, c='red', lw=2)

        plt.grid(True)


        plt.show()




def main():
    # Load the model configuration file and the data (observed cases)
     plot()


if __name__ == '__main__':
    # This script accepts two input argument:
    # 1) The path to the datafile to be postprocessed
    main()