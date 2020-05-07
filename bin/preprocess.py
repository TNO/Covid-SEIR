import sys
import os
from src.io_func import load_config
from bin.analyseIC import save_and_plot_IC
from src.api_nl_data import scrape_and_post_nl_data, get_and_save_nl_data


def preprocess(configpath, save_plots=True):
    # This function will
    #   1) scrape the data from the RIVM and NICE sites and post it to the database,
    #   2) pull from the database and generate the dataNL file and the icdata file (in the res directory), and
    #   3) generate the icufrac file and ICanalysis plots (in the output directory)

    config = load_config(configpath)
    output_base_filename = (os.path.split(configpath)[-1]).split('.')[0]

    # Check if the IC data file is defined in config, otherwise use '../res/icdata_main.txt'
    try:
        ic_data_file = config['icdatafile']
    except KeyError:
        print("No icdatafile in config, using '../res/icdata_main.txt'.")
        ic_data_file = '../res/icdata_main.txt'

    scrape_and_post_nl_data()
    get_and_save_nl_data(config['startdate'], config['country'], ic_data_file)
    save_and_plot_IC(config, output_base_filename, int(save_plots))
    print("Preprocessing complete.")


if __name__ == '__main__':
    # This script accepts two input arguments:
    # 1) the path to the configuration .json file
    # 2) the choice to save the plots (0 for no, 1 for yes)
    preprocess(sys.argv[1])
