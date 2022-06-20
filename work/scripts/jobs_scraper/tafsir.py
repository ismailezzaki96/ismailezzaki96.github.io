import requests
from bs4 import BeautifulSoup
import os

verbose = True
def getdata(url):
    r = requests.get(url)
    return r.text
# Get Html code using parse


def html_code(url):
    # pass the url
    # into getdata function
    htmldata = getdata(url)
    soup = BeautifulSoup(htmldata, 'html.parser')
    # return html code
    return(soup)

tafsir = "qotb"
base_url = "https://quran-tafsir.net"
aya= 1
sura= 1
while (sura < 315):
    os.makedirs(f"{tafsir}/sura{sura}", exist_ok = True)
    url = f"{base_url}/{tafsir}/sura{sura}/-aya{aya}.html"
    if verbose:
        print(url)
    r = requests.get(url)
    if (r.status_code == 200):
        html = r.text
        # replace the https:// link with a relative link to work offline
        html = html.replace(base_url, "")
        # create file and save the html code
        open(f"{tafsir}/sura{sura}/-aya{aya}.html","w").write(html)
        # to the next aya
        aya +=1
    else:
        sura += 1
        aya = 1
