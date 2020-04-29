import sys
from bin.analyseIC import plot_IC, load_config
from bin.retrieve_nl_data import retrieve_nl_data


def preprocess(configpath, icfile):
    plot_IC(configpath, icfile)
    config = load_config(configpath)
    retrieve_nl_data(config['startdate'])


if __name__ == '__main__':
    # This script only accepts one input argument: the path to the configuration .json file
    preprocess(sys.argv[1], sys.argv[2])
