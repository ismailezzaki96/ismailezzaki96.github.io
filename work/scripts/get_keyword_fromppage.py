import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

url = "https://www.doc-solus.fr/main.html?words=&filiere=--&matiere=Physique&concours=--&annee=--"

#If there is no such folder, the script will create one automatically
""" folder_location = "/home/k/Desktop/phd/Elementary Particle Theory/"
if not os.path.exists(folder_location):
    os.mkdir(folder_location)
"""
file = open("/home/k/Desktop/work/phd/physique_lexique.txt","w")

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
i=0
for link in soup.select("a"):
    print(link)
    response = requests.get("https://www.doc-solus.fr/" + link['href'])
    soup = BeautifulSoup(response.text, "html.parser")
    div = soup.find("pre", {"id": "enoncetxt"})
    if div is not None:
        file.write(div.text + "\n")