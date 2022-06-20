#!/usr/bin/env python
import time
import requests
from bs4 import BeautifulSoup


from selenium import webdriver

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
options.add_argument('--headless')
driver = webdriver.Chrome("/home/k/chromedriver", options=options)

URL = 'https://inspirehep.net/jobs?sort=mostrecent&size=100&page=1&q=&rank=PHD&rank=MASTER'

driver.get( URL)

time.sleep(5)

page_source = driver.page_source


soup = BeautifulSoup(page_source , 'html.parser')


job_elems = soup.find_all(class_='result-item-title')


for job_elem in job_elems:
    URL = 'https://inspirehep.net' + job_elem['href']
    driver.get(URL)
    time.sleep(5)
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    date = soup.find('div', class_='ant-row-end')
    end_date = date.find_all('span')
    if end_date[0] != end_date[1]:
        print(end_date[0] , '\t', end_date[1])
        print (URL)

