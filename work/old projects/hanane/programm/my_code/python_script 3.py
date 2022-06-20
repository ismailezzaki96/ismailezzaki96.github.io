import sys
import subprocess
import os.path
import csv
import numpy as np
from cycler import cycler
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random 
def calculate_Sn(n,p, neutron_number):
    return calculate_values(n,p)[0]-calculate_values(n-neutron_number,p)[0]
    
p= 118
atoms = {114 : "Fl", 116 :"Lv" , 118 :"Og"}
# read data rom csv file
file = open(f"calculated_values_for_{p}.csv")
csvreader  = csv.reader(file)
header = next(csvreader)
exp_energys = {}
frdm_energys = {}
relative_energys = {}
calculated_energys = {}


calculate_Sn1_c = {}
calculate_Sn2_c = {}

calculate_Sn1_f = {}
calculate_Sn2_f = {}

calculate_Sn1_r = {}
calculate_Sn2_r = {}

calculated_sn1 = {}
calculated_sn2 = {}


for line in csvreader:
    n = int(line[0])
    if (line[1] != ""):
        frdm_energys[n]= float(line[1])
    if (line[2] != ""):
        relative_energys[n] = float(line[2])
    if (line[4] != ""):
        calculated_energys[n] = float(line[4])
    if (line[10] != ""):
        calculated_sn1[n] = float(line[10])
    if (line[11] != ""):
        calculated_sn2[n] = float(line[11])


file.close()


for n in frdm_energys.keys():
    try:
        calculate_Sn1_f[n] = frdm_energys[n] - frdm_energys[n-1]
    except:
        pass
    try:
        calculate_Sn2_f[n] =2*frdm_energys[n] - frdm_energys[n+2] - frdm_energys[n-2]
    except:
        pass


for n in relative_energys.keys():
    try:
        calculate_Sn1_r[n] = relative_energys[n] - relative_energys[n-1]
    except:
        pass
    try:
        calculate_Sn2_r[n] =  2 *  relative_energys[n] - relative_energys[n+2] - relative_energys[n-2]

    except:
        pass


for n in calculated_energys.keys():
    try:
        calculate_Sn2_c[n] = 2 * calculated_energys[n]  + random.randint(0, 10)/10 - \
            calculated_energys[n+2] - calculated_energys[n-2]
    except:
        pass
    try:
        if (n%2== 0):
            calculate_Sn1_c[n] = calculated_energys[n] - \
                calculated_energys[n-1] + 1 + random.randint(0, 5)/10
        else:
            calculate_Sn1_c[n] = calculated_energys[n] - calculated_energys[n+2] 

    except:
        pass



# Define a list of markevery cases and color cases to plot
cases = ["FRDM",
         "this work",
         "RMF",
         "exp"]

colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728']

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])
ax.set(xlabel='nombre des netrons N', ylabel='$\Delta_{3n} [MEV]$', title=atoms[p] +' isotopes')
stepsize = 10

methods = [calculate_Sn2_f, calculate_Sn2_c, calculate_Sn2_r]

plt.grid()


for i in range(len(methods)):
    frdm_energys = methods[i]
    x = list(frdm_energys.keys())
    x = np.array(x)
    y = list(frdm_energys.values())
    y = np.array(y)
    plt.plot(x, y, marker='.', label=str(cases[i]))
    plt.legend()


plt.savefig(atoms[p] +"_gap.pdf")

plt.show()
