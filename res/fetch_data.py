import os

try:
    import urllib2
except ImportError:
    import urllib.request as urllib2

URL = 'https://coronavirus-tracker-api.herokuapp.com/all'

if os.path.split(os.getcwd())[1] == 'bin' or os.path.split(os.getcwd())[1] == 'vis':
    FILENAME = os.path.join(os.path.split(os.getcwd())[0],'res','covid-19_data.json')
else:
    FILENAME = 'res/covid-19_data.json'


def fetch():
    url = urllib2.urlopen(URL)

    r = url.read()

    print("read bytes from %s: %i" % (URL, len(r)))

    if len(r) < 1000:
        raise Exception("fetch_data.py read less than 1000 bytes")

    with open(FILENAME, 'wb') as f:
        f.write(r)


if __name__ == '__main__':
    fetch()
