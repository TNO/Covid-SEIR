from datetime import datetime
import os.path
import time
import numpy as np
import pandas as pd
import urllib.request as urllib

baseURL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
baseFS = os.path.dirname(os.path.abspath(__file__))
confirmed = 'time_series_covid19_confirmed_global.csv'
deaths = 'time_series_covid19_deaths_global.csv'
recovered = 'time_series_covid19_recovered_global.csv'
all = [confirmed, deaths, recovered]


def fetch_all():

    for f in all:
        fetch(baseURL, f)


def fetch(base, csv_file):

    URL = base+csv_file
    FILENAME = os.path.join(baseFS, csv_file)

    try:
        url = urllib.urlopen(URL)

        r = url.read()

        print("read bytes from %s: %i" % (URL, len(r)))

        if len(r) < 1000:
            raise Exception("global_data.py read less than 1000 bytes")

        with open(FILENAME, 'wb') as f:
            f.write(r)
    except Exception as e:
        if  'HTTP Error 503' in str(e):
            print('John Hopkins repository service unavailable: no new data loaded')
        else:
            raise e




def load_data():
    confirmedfile = os.path.join(baseFS, confirmed)

    CACHETIMESECONDS = 3600 * 3  # be nice to the API to not get banned

    if (not os.path.exists(confirmedfile) or os.path.getmtime(confirmedfile) < time.time() - CACHETIMESECONDS):
        fetch_all()

    c = pd.read_csv(os.path.join(baseFS, confirmed), delimiter=',', dtype=None)
    d = pd.read_csv(os.path.join(baseFS, deaths), delimiter=',', dtype=None)
    r = pd.read_csv(os.path.join(baseFS, recovered), delimiter=',', dtype=None)
    
    return c, d, r


def get_country_data(country, province=''):
    c, d, r = load_data()

    if province == '':
        sel_c = c.loc[(c['Country/Region'] == country) & (c['Province/State'].isnull())]
        sel_d = d.loc[(d['Country/Region'] == country) & (d['Province/State'].isnull())]
        sel_r = r.loc[(r['Country/Region'] == country) & (r['Province/State'].isnull())]
    elif province == 'all':
        sel_c = c.loc[(c['Country/Region'] == country) & (c['Province/State'].notnull())]
        sel_d = d.loc[(d['Country/Region'] == country) & (d['Province/State'].notnull())]
        sel_r = r.loc[(r['Country/Region'] == country) & (r['Province/State'].notnull())]
    else:
        sel_c = c.loc[(c['Country/Region'] == country) & (c['Province/State'] == province)]
        sel_d = d.loc[(d['Country/Region'] == country) & (d['Province/State'] == province)]
        sel_r = r.loc[(r['Country/Region'] == country) & (r['Province/State'] == province)]

    dates = [datetime.strptime(d, "%m/%d/%y") for d in sel_c.columns[4:]]
    firstdate = dates[0]

    days = [(d-firstdate).days+1 for d in dates]

    if len(sel_c) == 0:
        print(f'no data in dataset for {country} {province}')
        return None
    elif len(sel_c) > 1:
        c_data = sel_c.sum().to_numpy()[4:]
        d_data = sel_d.sum().to_numpy()[4:]
        r_data = sel_r.sum().to_numpy()[4:]
    else:
        c_data = sel_c.to_numpy()[0, 4:]
        d_data = sel_d.to_numpy()[0, 4:]
        r_data = sel_r.to_numpy()[0, 4:]

    data = np.stack((days, c_data, d_data, r_data), axis=-1)

    return data, firstdate


if __name__ == '__main__':

    print(get_country_data('China', 'all'))
