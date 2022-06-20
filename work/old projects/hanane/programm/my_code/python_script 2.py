import sys
import subprocess
import os.path
import csv
import numpy as np
from cycler import cycler
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def calculate_Sn(n,p, neutron_number):
    return calculate_values(n,p)[0]-calculate_values(n-neutron_number,p)[0]

p= 114
# read data rom csv file
file = open(f"calculated_values_for_{p}.csv")
csvreader  = csv.reader(file)
header = next(csvreader)
exp_energys = {}
frdm_energys = {}
relative_energys = {}
calculated_energys = {}
calculated_pairing_energy = {}
calculated_Rn = {}
calculated_Rp = {}
calculate_Rc = {}
calculate_Sn2 = {}

for line in csvreader:
    n = int(line[0])
    calculated_Rn[n] = float(line[8])
    calculated_Rp[n] = float(line[9])
    calculate_Rc[n] =  np.sqrt( calculated_Rp[n]**2 + 0.64)


file.close()
print(calculated_Rp)


# Define a list of markevery cases and color cases to plot
cases = ["FDM",
         "HFOOD",
         "RC-HBT",
         "EXP"]

colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728']

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
ax.set(xlabel='nombre des netrons N', ylabel='BE/A [MEV]', title='Fl isotopes')

methods = [calculated_Rn , calculated_Rp ]


for i in range(len(methods)):
    frdm_energys = methods[i]
    x= list(frdm_energys.keys())
    x = np.array(x)
    y = list(frdm_energys.values())
    y = np.array(y)
    ax.plot(x,y, label=str(cases[i]))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


plt.show()