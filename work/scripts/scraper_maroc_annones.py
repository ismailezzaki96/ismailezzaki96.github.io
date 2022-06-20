#!/usr/bin/env python 
import time
import requests
from bs4 import BeautifulSoup
import os
import subprocess
import validators
import sys

""" 
from selenium import webdriver

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
options.add_argument('--headless')
driver = webdriver.Chrome(
    "/home/k/Desktop/python scripts/chromedriver", chrome_options=options)

URL = 'https://inspirehep.net/jobs?sort=mostrecent&size=100&page=1&q=&rank=PHD&rank=MASTER'

driver.get( URL)

time.sleep(5)

page_source = driver.page_source """


def is_valid_url(url):
    import re
    regex = re.compile(
        r'^https?://'  # http:// or https://
        # domain...
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url is not None and regex.search(url)

def get_annones(url):
    if  not validators.url(url):
        print (url)
        sys.exit("invalid url")
 #   print ("base_url " + url)
    file = open('annonnes.txt', 'a')
    page_source = requests.get(url)
    soup = BeautifulSoup(page_source.text, 'html.parser')
    soup = soup.find(class_="cars-list")
    titles  = soup.find_all("li")
    i = 0
    answer = ""
    date = ""
    description = ""
    for job_elem in titles:
        if (job_elem.find(class_='time') != None):
            date = job_elem.find(class_='time')
            if ("Hier" in date.text):
                URL = job_elem.find("a").get("href")
                answer = input("do you want to ckeck  "  + job_elem.find(class_='holder').find("h3").text)
                if (answer ==  'y'):
                    page_source = requests.get("https://www.marocannonces.com/"+URL)
                   # print("https://www.marocannonces.com/"+  URL)
                    soup = BeautifulSoup(page_source.text, 'html.parser')
                    description = soup.find_all(class_="block")[1].text
                    print (description)
                    answer = input("do you want to apply?\n")
                    if (answer == 'y'):
                        URL = "https://www.marocannonces.com/"+URL
                        subprocess.run(["open", URL])
                        answer = input("next ?\n")
                       # os.system( "open " + URL)
            i=i+1
    file.close()
    return(date.text)


def prepare_urls(url):
    last_item = 'Hier'
    i = 2
    while ("Hier" in last_item or "Aujourd'hui" in last_item):
        last_item = get_annones(url)
    #   print (last_item)
        url = url + "?pge=" + str(i)
        i = +1


""" file = open('annonnes.txt', 'w')
file.write("")
file.close() """

urls = ("https://www.marocannonces.com/maroc/offres-emploi-rabat-b309-t590.html", "https://www.marocannonces.com/maroc/offres-emploi-safi-b309-t591.html"
           )


for url in urls:
    prepare_urls(url)
