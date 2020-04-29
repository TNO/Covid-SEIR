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

date_conversion = {'jan': 'Jan', 'feb': 'Feb', 'mrt': 'Mar', 'apr': 'Apr', 'mei': 'May', 'jun': 'Jun',
                   'jul': 'Jul', 'aug': 'Aug', 'sep': 'Sep', 'okt': 'Oct', 'nov': 'Nov', 'dec': 'Dec'}


def retrieve_nl_data(config_start_date):
    # Get RIVM data from charts
    options = Options()
    options.headless = True
    options.setLogLevel = 'OFF'
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

    # Get NICE JSON data
    nice_html = requests.get('https://www.stichting-nice.nl/covid-19/public/intake-count/')
    nice_data_list = json.loads(nice_html.content)
    nice_data = np.zeros(len(nice_data_list), dtype={'names': ('date', 'value'), 'formats': ('datetime64[D]', '<i4')})
    nice_data['date'] = np.asarray([data_entry['date'] for data_entry in nice_data_list], dtype='datetime64[D]')
    nice_data['value'] = np.asarray([data_entry['value'] for data_entry in nice_data_list], dtype='<i4')

    # Figure out the start date based on how many extra days the data is available for.
    rivm_days_extra = (infected_data['date'][0].astype(datetime) - datetime.strptime(
        config_start_date, '%m/%d/%y').date()).days
    nice_days_extra = (nice_data['date'][0].astype(datetime) - datetime.strptime(
        config_start_date, '%m/%d/%y').date()).days

    # Choose an end date based on who has data available. The end date will be constrained based on the minimum date of
    # the data.
    end_date = min(infected_data['date'][-1].astype(datetime), nice_data['date'][-1].astype(datetime))
    end_rivm_index = np.where(infected_data['date'].astype(datetime) == end_date)[0][0]
    end_nice_index = np.where(nice_data['date'].astype(datetime) == end_date)[0][0]
    data_matrix = np.vstack((
        np.arange(1, len(nice_data[-nice_days_extra:end_rivm_index + 1]) + 1),
        infected_data['cum'][-rivm_days_extra:end_rivm_index + 1],
        dead_data['cum'][-rivm_days_extra:end_rivm_index + 1],
        np.zeros(len(nice_data[-nice_days_extra:end_nice_index + 1])),
        hospitalized_data['cum'][-rivm_days_extra:end_rivm_index + 1],
        nice_data['value'][-nice_days_extra:end_nice_index + 1],
        np.zeros(len(nice_data[-nice_days_extra:end_nice_index + 1])))).T

    # Delete the last row if there is no new data
    if np.all(data_matrix[-1, 1:] == np.zeros(6)):
        data_matrix = data_matrix[:-1, :]

    # Set the output data file and save the data
    data_file = '../res/corona_dataNL_main.txt'
    np.savetxt(data_file, data_matrix, delimiter='\t', fmt='%d')


if __name__ == '__main__':
    # Input argument is the start date in this format (m/d/y): 3/1/20
    retrieve_nl_data(sys.argv[1])
