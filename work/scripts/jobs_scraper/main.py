#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
import validators
import sys

import dominate
from dominate.tags import *

from selenium import webdriver

from http.server import HTTPServer, BaseHTTPRequestHandler
import pickle
import inspect
import time


options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--incognito')
options.add_argument('--headless')
prefs = {"profile.managed_default_content_settings.images": 2}
options.add_experimental_option("prefs", prefs)
driver = webdriver.Chrome("/home/k/chromedriver", options=options)


city = 'rabat'
today = date.today().strftime("%d/%m/%Y")
yesterday = (date.today() + timedelta(days=-1)).strftime("%d/%m/%Y")
# print(today)
# print(yesterday)

doc = dominate.document(title='Dominate your HTML')

with doc.head:
    link(rel='stylesheet', href='https://cdn.jsdelivr.net/npm/water.css@2/out/dark.css')
    link(rel='stylesheet', href='style.css')
    meta(content='text/html;charset=utf-8',  http_equiv='Content-Type')
    meta(content="utf-8",  http_equiv="encoding")


# user define function
# Scrape the data
# and get in string
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


def html_code_with_SL(url, methond=""):
    driver.get(url)
    if (methond == 'very_late'):
        time.sleep(5)
    page_state = driver.execute_script('return document.readyState;')
    while(page_state != 'complete'):
        page_state = driver.execute_script('return document.readyState;')
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")
    return (soup)


def get_last_element(name):
    try:
        file_obj = open("data.txt", "rb")
        dic = pickle.load(file_obj)
        file_obj.close()
    except:
        dic = {}
    last_item = dic.get(name, "")
    return last_item


def set_last_element(last_item, name):
    try:
        dic = pickle.load(open("data.txt", "rb"))
    except:
        dic = {}
    dic[name] = last_item
    dic_file = open("data.txt", "wb")
    pickle.dump(dic, dic_file)


# alwadifa
def alwadifa():
    name = inspect.stack()[0][3]
    last_item = get_last_element(name)
    doc.add(h2("al wadifa maroc"))
    list = ul()
    url = "http://www.alwadifa-maroc.com"
    soup = html_code(url)
    job_elems = soup.find_all("div", {"class": "bloc-content"})
    for elem in job_elems:
        date = elem.find("li")
        if (date.text == today or date.text == yesterday):
            elem = elem.find("a")
            title = elem.text
            link = url+elem["href"]
            if (link == last_item):
                list += li("---------------")
            list += li(a(title, href=link), __pretty=False)
    last_item = url+job_elems[0].find("a")["href"]
    set_last_element(last_item, name)
    doc.add(list)

# emploi


def emploi():
    name = inspect.stack()[0][3]
    last_item = get_last_element(name)
    link = ""

    doc.add(h2("emploi maroc"))
    base_url = "https://www.emploi.ma"
    url = base_url+"/recherche-jobs-maroc/?f%5B0%5D=im_field_offre_region%3A64"
    list = ul()
    soup = html_code(url)
    job_elems = soup.select(".job-description-wrapper")
    link = ""
    for elem in job_elems:
        date = elem.select_one(".job-recruiter").text
        date = date.split(" ")[0]
        date = date.replace(".", "/")
        if (date == today or date == yesterday):
            elem = elem.select_one("h5 a")
            title = elem.text
            link = elem["href"]
            if (link == last_item):
                list += li("---------------")
            list += li(a(title, href=base_url + link), __pretty=False)
    last_item = base_url + job_elems[0].select_one("h5 a")["href"]
    set_last_element(last_item, name)
    doc.add(list)

# maroc annonce


def get_annones(url):
    list = ul()
    if not validators.url(url):
        print(url)
        sys.exit("invalid url")
    soup = html_code(url)
    soup = soup.find(class_="cars-list")
    titles = soup.find_all("li")
    answer = ""
    date = ""
    description = ""
    for job_elem in titles:
        if (job_elem.find(class_='time') != None):
            date = job_elem.find(class_='time')
            if ("Hier" in date.text):
                URL = job_elem.find("a").get("href")
                link = "https://www.marocannonces.com/" + URL
                title = job_elem.find(class_='holder').find("h3").text
                list += li(a(title, href=link), __pretty=False)
    doc.add(list)
    return(date.text)


def marocannonces(base_url):
    doc.add(h2("maroc annonce"))
    last_item = 'Hier'
    i = 2
    url = base_url
    while ("Hier" in last_item or "Aujourd'hui" in last_item):
        last_item = get_annones(url)
        url = base_url + "?pge=" + str(i)
        i += 1
        print(i)


# indeed
def indeed():
    name = inspect.stack()[0][3]
    last_item = get_last_element(name)
    link = ""
    doc.add(h2("indeed"))
    list = ul()
    location = "Rabat"
    base_url = "https://ma.indeed.com"
    url = base_url + "/jobs?q&l=" + location + "&radius=50&fromage=3&limit=50"
    soup = html_code(url)
    link = ""
    job_elems = soup.find_all('a', class_="tapItem")
    for elem in job_elems:
        title = elem.select('.jobTitle span')[1].text
        link = base_url + elem["href"]
        if (link == last_item):
            list += li("---------------")
        list += li(a(title, href=link), __pretty=False)
    last_item = base_url + job_elems[0]["href"]
    set_last_element(last_item, name)
    doc.add(list)


def extrat_positon(positions):
    list = ul()
    keywords = ["PHD", "PhD", "Ph.D", " Doctoral ", "Graduate", " Doctorate "]
    for position in positions:
        is_phd = False
        title_and_link = position.find(class_="col-sm-12")
        title = title_and_link.text
        for keyword in keywords:
            if keyword in title:
                is_phd = True
        if is_phd:
            link = title_and_link.find('a')["href"]
            list += li(a(title, href="https://euraxess.ec.europa.eu" +
                         link), __pretty=False)
    doc.add(list)

# euraxess


def euraxess(keyword):
    doc.add(h2("euraxess: " + keyword))
    i = 1
    base_url = "https://euraxess.ec.europa.eu/jobs/search/field_research_field/astronomy-33/field_research_field/physics-344/field_research_profile/first-stage-researcher-r1-446?keywords=" + keyword
    URL = base_url
    while (1):
        soup = html_code_with_SL(URL)
        positions = soup.find_all("div", {"class": "views-row"})
        if (positions == []):
            break
        extrat_positon(positions)
        URL = base_url + "&page="+str(i)
        i += 1
        print(i)

# academic jobs


def academic():
    base_url = "https://academicjobsonline.org"
    url = base_url + "/ajo?joblist-0-0-0-0------"
    soup = html_code(url)
    doc.add(h2("academic"))
    list = ul()
    job_elems = soup.select("ol li")
    keywords = ["PHD", "PhD", "Ph.D", " Doctoral ", "Graduate", " Doctorate "]
    for elem in job_elems:
        title = elem.text
        if any(key in title for key in keywords):
            print(title)
            link = base_url + elem.select_one("a")["href"]
            print(elem.select_one("a"))
            list += li(a(title, href=link), __pretty=False)
    doc.add(list)


# inspire
def inspire():
    name = inspect.stack()[0][3]
    base_url = "https://inspirehep.net"
    URL = base_url + '/jobs?sort=mostrecent&size=100&page=1&q=&rank=PHD&rank=MASTER'
    soup = html_code_with_SL(URL, methond="very_late")
    job_elems = soup.find_all(class_='mv2')
    doc.add(h2("inspire"))
    list = ul()
    for elem in job_elems:
        date = elem.find(class_='ant-row-space-between')
        old_dates = ["days", "year", "month"]
        if not any(old_date in date.text for old_date in old_dates):
            # print(date.text)
            position = elem.find(class_='result-item-title')
            title = position.text
            link = base_url + position["href"]
            list += li(a(title, href=link), __pretty=False)
    doc.add(list)

# rekrute


def rekrute():
    name = inspect.stack()[0][3]
    last_item = get_last_element(name)
    link = ""
    doc.add(h2("rekrute"))
    base_url = "https://www.rekrute.com"
    list = ul()
    i = 1
    more = True
    while(more):
        url = base_url + "/offres.html?s=3&p=" + \
            str(i)+"&o=1&positionId%5B0%5D=13&regionId=12"
        soup = html_code(url)
        job_elems = soup.select(".post-id")
        if len(job_elems) == 0:
            break
        for elem in job_elems:
            date = elem.select_one(".date span").text
            print(date)
            if (date != today and date != yesterday):
                more = False
                break
            elem = elem.select_one("h2 a")
            title = elem.text
            link = elem["href"]
            list += li(a(title, href=base_url + link), __pretty=False)
        i += 1
        print(i)

    set_last_element(link, name)
    doc.add(list)


def anapec():
    doc.add(h2("anapec"))
    list = ul()
    i = 1
    base_url = "http://anapec.org"
    more = True
    while(more):
        url = base_url + "/sigec-app-rv/chercheurs/resultat_recherche/page:" + \
            str(i)+"/region:4/language:fr"
        soup = html_code(url)
        job_elems = soup.select(".tablesorter .header tr")
        for elem in job_elems:
            date = elem.select("td")[2].text
            if (date != today or date != yesterday):
                more = False
            location = elem.select("td")[-1].text
            if city in location.lower():
                link = elem.select_one("td a")["href"]
                title = elem.select("td")[3].text
                list += li(a(title, href=base_url + link), __pretty=False)
        i += 1
    doc.add(list)


#print("alwadifa")
#alwadifa()
print("indeed")
indeed()

print("earaxess 1")
#euraxess("computational%20physics")
#print("earaxess 2")
#euraxess("%20particle%20physics")

#print("inspire")
#inspire()

print("rekrute")
rekrute()

print("marocannonces")
url = ("https://www.marocannonces.com/maroc/offres-emploi-rabat-b309-t590.html")
marocannonces(url)

print("emploi")
emploi()


print("anapec")
anapec()

file = open( "jobs.html", "w")

file.write(doc.render())
file.close()


class Serv(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = 'jobs.html'
        self.path = self.path.replace('/','')
        try:
            file_to_open = open(self.path).read()
            self.send_response(200)
        except:
            print(self.path)
            file_to_open = self.path
            self.send_response(404)
        self.end_headers()
        self.wfile.write(bytes(file_to_open, 'utf-8'))


print("open http://localhost:8080 to view available positions")
httpd = HTTPServer(('localhost', 8080), Serv)
httpd.serve_forever()
