import sys
import os
import requests
import lxml.html
import json
import numpy as np
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import re
from src.io_func import load_config

date_conversion = {'jan': 'Jan', 'feb': 'Feb', 'mrt': 'Mar', 'apr': 'Apr', 'mei': 'May', 'jun': 'Jun',
                   'jul': 'Jul', 'aug': 'Aug', 'sep': 'Sep', 'okt': 'Oct', 'nov': 'Nov', 'dec': 'Dec'}


def scrape_nl_data():
    print("Scraping RIVM data...")
    # Get RIVM data from charts
    options = Options()
    options.headless = True
    driver = webdriver.Firefox(options=options, executable_path=[os.path.dirname(os.getcwd()) + '/res/geckodriver.exe'],
                               service_log_path='../output/geckodriver.log')
    driver.get('https://www.rivm.nl/coronavirus-covid-19/grafieken')

    try:
        WebDriverWait(driver, 10).until(EC.visibility_of_element_located((
            By.XPATH, '//*[name()="g" and @class="highcharts-series-group"]')))
    except TimeoutException:
        print('Timed out waiting for page to load')
        driver.quit()
        sys.exit(1)

    # Get the charts from RIVM
    [infected_chart, hospitalized_chart, dead_chart, dead_age_gender] = driver.find_elements_by_xpath(
        '//*[name()="g" and @class="highcharts-series-group"]')

    # Infected data from RIVM
    infected_new_html = infected_chart.find_elements_by_xpath(
        './/*[name()="rect" and @class="highcharts-point highcharts-color-0"]')
    infected_existing_html = infected_chart.find_elements_by_xpath(
        './/*[name()="rect" and @class="highcharts-point highcharts-color-1"]')
    infected_new_text = [re.split(' |\.|,', value.get_attribute('aria-label')) for value in infected_new_html]
    infected_existing_text = [re.split(' |\.|,', value.get_attribute('aria-label')) for value in infected_existing_html]
    date_list = [datetime.strptime(item[2] + '/' + date_conversion[item[3]] + '/20', '%d/%b/%y')
                 for item in infected_existing_text]
    infected_data = np.zeros(len(infected_new_text), dtype={'names': ('index', 'date', 'new', 'existing', 'cum'),
                                                            'formats': ('<i4', 'datetime64[D]', '<i4', '<i4', '<i4')})
    infected_data['index'] = np.asarray(infected_new_text)[:, 0]
    infected_data['date'] = np.asarray(date_list)
    infected_data['new'] = np.asarray(infected_new_text)[:, 5]
    infected_data['existing'] = np.asarray(infected_existing_text)[:, 5]
    infected_data['cum'] = np.cumsum(infected_data['new'] + infected_data['existing'])

    # Hospitalized data from RIVM
    hospitalized_new_html = hospitalized_chart.find_elements_by_xpath(
        './/*[name()="rect" and @class="highcharts-point highcharts-color-0"]')
    hospitalized_existing_html = hospitalized_chart.find_elements_by_xpath(
        './/*[name()="rect" and @class="highcharts-point highcharts-color-1"]')
    hospitalized_new_text = [re.split(' |\.|,', value.get_attribute('aria-label')) for value in hospitalized_new_html]
    hospitalized_existing_text = [
        re.split(' |\.|,', value.get_attribute('aria-label')) for value in hospitalized_existing_html]
    date_list = [datetime.strptime(item[2] + '/' + date_conversion[item[3]] + '/20', '%d/%b/%y')
                 for item in hospitalized_existing_text]
    hospitalized_data = np.zeros(len(hospitalized_new_text),
                                 dtype={'names': ('index', 'date', 'new', 'existing', 'cum'),
                                        'formats': ('<i4', 'datetime64[D]', '<i4', '<i4', '<i4')})
    hospitalized_data['index'] = np.asarray(hospitalized_new_text)[:, 0]
    hospitalized_data['date'] = np.asarray(date_list)
    hospitalized_data['new'] = np.asarray(hospitalized_new_text)[:, 5]
    hospitalized_data['existing'] = np.asarray(hospitalized_existing_text)[:, 5]
    hospitalized_data['cum'] = np.cumsum(hospitalized_data['new'] + hospitalized_data['existing'])

    # Death data from RIVM
    dead_new_html = dead_chart.find_elements_by_xpath(
        './/*[name()="rect" and @class="highcharts-point highcharts-color-0"]')
    dead_existing_html = dead_chart.find_elements_by_xpath(
        './/*[name()="rect" and @class="highcharts-point highcharts-color-1"]')
    dead_new_text = [re.split(' |\.|,', value.get_attribute('aria-label')) for value in dead_new_html]
    dead_existing_text = [re.split(' |\.|,', value.get_attribute('aria-label')) for value in dead_existing_html]
    date_list = [datetime.strptime(item[2] + '/' + date_conversion[item[3]] + '/20', '%d/%b/%y')
                 for item in dead_existing_text]
    dead_data = np.zeros(len(dead_new_text), dtype={'names': ('index', 'date', 'new', 'existing', 'cum'),
                                                    'formats': ('<i4', 'datetime64[D]', '<i4', '<i4', '<i4')})
    dead_data['index'] = np.asarray(dead_new_text)[:, 0]
    dead_data['date'] = np.asarray(date_list)
    dead_data['new'] = np.asarray(dead_new_text)[:, 5]
    dead_data['existing'] = np.asarray(dead_existing_text)[:, 5]
    dead_data['cum'] = np.cumsum(dead_data['new'] + dead_data['existing'])

    driver.quit()

    print("Scraping NICE data...")

    # Get NICE JSON data
    icu_cum_html = requests.get('https://www.stichting-nice.nl/covid-19/public/intake-cumulative')
    icu_cum_data_list = json.loads(icu_cum_html.content)
    icu_dead_cum_html = requests.get('https://www.stichting-nice.nl/covid-19/public/died-and-survivors-cumulative')
    icu_dead_cum_data_list = json.loads(icu_dead_cum_html.content)[0]
    icu_rec_cum_data_list = json.loads(icu_dead_cum_html.content)[2]
    icu_hos_rec_cum_data_list = json.loads(icu_dead_cum_html.content)[1]
    icu_pres_html = requests.get('https://www.stichting-nice.nl/covid-19/public/intake-count')
    icu_pres_data_list = json.loads(icu_pres_html.content)

    # Arrange NICE data
    nice_data = np.zeros(len(icu_pres_data_list), dtype={
        'names': ('date', 'icu_cum', 'icu_dead_cum', 'icu_rec_cum', 'icu_hos_rec_cum', 'icu_pres'),
        'formats': ('datetime64[D]', '<i4', '<i4', '<i4', '<i4', '<i4')})
    nice_data['date'] = np.asarray([data_entry['date'] for data_entry in icu_pres_data_list], dtype='datetime64[D]')
    nice_data['icu_cum'] = np.asarray([data_entry['value'] for data_entry in icu_cum_data_list], dtype='<i4')
    nice_data['icu_dead_cum'] = np.asarray([data_entry['value'] for data_entry in icu_dead_cum_data_list], dtype='<i4')
    nice_data['icu_rec_cum'] = np.asarray([data_entry['value'] for data_entry in icu_rec_cum_data_list], dtype='<i4')
    nice_data['icu_hos_rec_cum'] = np.asarray(
        [data_entry['value'] for data_entry in icu_hos_rec_cum_data_list], dtype='<i4')
    nice_data['icu_pres'] = np.asarray([data_entry['value'] for data_entry in icu_pres_data_list], dtype='<i4')

    # Arrange all the data needed from NICE and RIVM
    extra_empty_values = len(icu_pres_data_list) - len(infected_data)
    all_data = np.zeros(len(icu_pres_data_list), dtype={
        'names': ('date', 'inf_cum', 'hos_cum', 'dead_cum', 'icu_cum', 'icu_dead_cum',
                  'icu_rec_cum', 'icu_hos_rec_cum', 'icu_pres'),
        'formats': ('datetime64[D]', '<i4', '<i4', '<i4', '<i4', '<i4', '<i4', '<i4', '<i4')})
    all_data['date'] = nice_data['date']
    all_data['inf_cum'] = np.hstack((infected_data['cum'], np.zeros(extra_empty_values)))
    all_data['hos_cum'] = np.hstack((hospitalized_data['cum'], np.zeros(extra_empty_values)))
    all_data['dead_cum'] = np.hstack((dead_data['cum'], np.zeros(extra_empty_values)))
    all_data['icu_cum'] = nice_data['icu_cum']
    all_data['icu_dead_cum'] = nice_data['icu_dead_cum']
    all_data['icu_rec_cum'] = nice_data['icu_rec_cum']
    all_data['icu_hos_rec_cum'] = nice_data['icu_hos_rec_cum']
    all_data['icu_pres'] = nice_data['icu_pres']

    return all_data


def post_nl_data_to_db(data):
    print('Posting data to database...')
    dict_data = [{'date': str(item['date']), 'inf_cum': int(item['inf_cum']), 'hos_cum': int(item['hos_cum']),
                  'dead_cum': int(item['dead_cum']), 'icu_cum': int(item['icu_cum']),
                  'icu_dead_cum': int(item['icu_dead_cum']), 'icu_rec_cum': int(item['icu_rec_cum']),
                  'icu_hos_rec_cum': int(item['icu_hos_rec_cum']), 'icu_pres': int(item['icu_pres'])}
                 for item in data]
    payload = dict(enumerate(dict_data))
    requests.post('https://rex-co2.eu/api/post-corona-nl-data', json=payload)


def get_nl_db_data():
    print("Retrieving data from database...")
    # Get NL JSON data
    nl_data_html = requests.get('https://rex-co2.eu/api/corona-nl-data')
    nl_data_list = json.loads(nl_data_html.content)
    names = ['index', 'date', 'inf_cum', 'hos_cum', 'dead_cum', 'icu_cum', 'icu_dead_cum', 'icu_rec_cum',
             'icu_hos_rec_cum', 'icu_pres']
    formats = ['<i4', 'datetime64[D]', '<i4', '<i4', '<i4', '<i4', '<i4', '<i4', '<i4', '<i4']
    nl_data = np.zeros(len(nl_data_list), dtype={'names': names,
                                                 'formats': formats})
    for i, name in enumerate(names):
        nl_data[name] = np.asarray([data_entry[name] for data_entry in nl_data_list], dtype=formats[i])

    # If there is no RIVM data for the last row, then exclude that row
    if nl_data['inf_cum'][-1] == 0:
        nl_data = nl_data[:-1]

    return nl_data


def save_data(start_date, data_file, ic_data_file, data):
    print("Saving data to files...")

    days_extra = (data['date'][0].astype(datetime) - datetime.strptime(start_date, '%m/%d/%y').date()).days
    # Prepare the data matrix with the following columns:
    #   - 0: day (number starting from 1)
    #   - 1: cumulative registered infected (positive test cases)
    #   - 2: cumulative dead
    #   - 3: cumulative recovered
    #   - 4: cumulative hospitalized
    #   - 5: actual IC units used (may be estimated or 0)
    #   - 6: actual hospitalized (put all to 0 to overwrite from estimates calculated from the hospital flow model)
    data_matrix = np.vstack((
        np.arange(1, len(data[-days_extra:]) + 1),
        data['inf_cum'][-days_extra:],
        data['dead_cum'][-days_extra:],
        np.zeros(len(data[-days_extra:])),
        data['hos_cum'][-days_extra:],
        data['icu_pres'][-days_extra:],
        np.zeros(len(data[-days_extra:]))
    )).T

    # Delete the last row if there is no new RIVM data
    # if np.all(data_matrix[-1, 1:5] == np.zeros(4)):
    #     data_matrix = data_matrix[:-1, :]

    # Prepare the ICU data matrix with the following columns:
    #   - 0: day (number starting from 1)
    #   - 1: cumulative IC patients
    #   - 2: cumulative IC patients that have died
    #   - 3: cumulative IC patients that have left the IC (but are still in the hospital)
    #   - 4: total IC patients
    #   - 5: cumulative IC patients that have left the hospital
    #   - 6: cumulative hospitalized patients (note: this is from the RIVM data)
    ic_headings = 'day\ticucum\ticudead\ticurec\ticupres\thosprec\thoscum'
    ic_data = np.vstack((
        np.arange(1, len(data[-days_extra:]) + 1),
        data['icu_cum'][-days_extra:],
        data['icu_dead_cum'][-days_extra:],
        data['icu_rec_cum'][-days_extra:],
        data['icu_pres'][-days_extra:],
        data['icu_hos_rec_cum'][-days_extra:],
        data['hos_cum'][-days_extra:],
    )).T

    # Save the data
    np.savetxt(data_file, data_matrix, delimiter='\t', fmt='%d')
    np.savetxt(ic_data_file, ic_data, delimiter='\t', fmt='%d', header=ic_headings, comments='')


def get_and_save_nl_data(startdate, data_file, ic_data_file):
    # Calls the NL data from the API (does not scrape the sites)
    nl_data = get_nl_db_data()
    # Save the corona data text file and the ic data text file (NOT the icufrac)
    save_data(startdate, data_file, ic_data_file, nl_data)


def scrape_and_post_nl_data():
    # Scrape the NL data from RIVM and NICE
    nl_data = scrape_nl_data()
    # Post the updated NL data to the database
    post_nl_data_to_db(nl_data)


def main(action, configpath):
    if action == 1:
        scrape_and_post_nl_data()
    elif action == 2:
        config = load_config(configpath)

        # Check if the IC data file is defined in config, otherwise use '../res/icdata_main.txt'
        try:
            ic_data_file = config['icdatafile']
        except KeyError:
            print("No icdatafile in config, using '../res/icdata_main.txt'.")
            ic_data_file = '../res/icdata_main.txt'

        # Input argument is the path to the config file
        get_and_save_nl_data(config['startdate'], config['country'], ic_data_file)


if __name__ == '__main__':
    # Input arguments are:
    #   1) choice of action (either 1 or 2):
    #      - 1 to scrape and save the data to the database
    #      - 2 to pull the data from the database and save the dataNL and icdata files
    #   2) the configpath

    main(int(sys.argv[1]), sys.argv[2])
