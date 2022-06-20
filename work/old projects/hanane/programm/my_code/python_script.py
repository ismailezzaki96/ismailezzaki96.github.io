import sys
import subprocess
import os.path
import csv
import numpy as np
from cycler import cycler
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def set_parameters(n,p,vpair= -512.95 ,deformation="0"):
    text =""
    with open("hfbtho_NAMELIST_original.dat") as file:
        text = file.read()
#        if n % 2 == 1:
            #n-=1
            #text=text.replace("neutron_blocking = 0, 0, 0, 0, 0", "neutron_blocking = 1, 0, 0, 0, 0")
        text = text.replace("proton_number = 100", "proton_number = "+ str(p)).replace("neutron_number = 100", "neutron_number = " + str(n)).replace("-2000.0",str(vpair)).replace("-555", deformation)
    input_file = open("hfbtho_NAMELIST.dat","w")
    input_file.write(text)
    input_file.close()

def run_HFBTHO(n,p):
    #subprocess.call(["rm", "hfbtho_output.hel"])
    file = open("./out.txt","w")
    subprocess.call(["./hfbtho_main", "<" , "/dev/null" , ">&" , "[output_file]"],stdout=file, shell = False)
    subprocess.call(["cp", "thoout.dat" , "./results/output_"+str(p)+"_"+str(n)])

def read_values(path):
    with open(path, "r") as file:
        for line in file:
            if ("tEnergy: ehfb (qp)..." in line):
                BE = -1 * float(line.split()[-1])
            if ("pairing energy" in line):
                PE =-1* float(line.split()[-1])
            if ("rms-radius" in line):
                Rn = float(line.split()[2])
                Rp = float(line.split()[3])
    return BE, PE, Rn ,Rp

def calculate_values(n,p,vpair= -512.95 ,deformation="0"):
    path = "./results/output_"+str(p)+"_"+str(n)
    if( os.path.isfile(path)):
        return read_values(path)
    else:
        set_parameters(n,p,vpair,deformation)
        run_HFBTHO(n,p)
        return read_values(path)


def calculate_Sn(n,p, neutron_number):
    return calculate_values(n,p)[0]-calculate_values(n-neutron_number,p)[0]
    
p= 118
title = "Og isotopes"
# read data rom csv file
file = open(f"data_{p}.csv")
csvreader  = csv.reader(file)
header = next(csvreader)
exp_energys = {}
frdm_energys = {}
relative_energys = {}

for line in csvreader:
    n = int(line[0])
    if (not line[3] == ""):
        exp_energys[n] = float(line[3])
    if (not line[1] == ""):
        frdm_energys[n] = float(line[1])
    if (not line[2] == ""):
        relative_energys[n] = float(line[2])

file.close()


calculated_energys = {}
calculated_pairing_energy = {}
calculated_Rn = {}
calculated_Rp = {}
calculate_Sn1 = {}
calculate_Sn2 = {}

#deformation_state =  ["0.3","0.2","0.1","0","-0.1","-0.2" ,"-0.3" ]
deformation_state = "0"
all_keys = list(relative_energys.keys()) + list(frdm_energys.keys())
n_min = min(all_keys)
n_max = max(all_keys)
for n in range(n_min , n_max + 1):
    best_vals = None
    exp_energy = 0

    # find the best vpair only in the first run for n = 170
    step = 0.5
    # vpair  = -512.95
    # vpair = -515
    vpair  = -517
#    for vpair in np.arange(-515,-514,step):
        #print("the vpair is ", vpair, "step",step)
        # find the best state
        #for state in deformation_state:
    vals_for_state= calculate_values(n, p, vpair, deformation_state)
    energy = vals_for_state[0]
        #if (best_vals is None):
            #best_vals= vals_for_state
        #elif (abs(best_vals[0]- exp_energy) > abs(vals_for_state[0]- exp_energy) ):
            #best_vals = vals_for_state
    calculated_energys[n]=  energy
    calculated_pairing_energy[n] = vals_for_state[1]
    calculated_Rn[n] =  vals_for_state[2]
    calculated_Rp[n]=  vals_for_state[3]
    calculate_Sn1[n] = calculate_Sn(n,p,1)
    calculate_Sn2[n] = calculate_Sn(n,p,2)

"""
file = open(f"calculated_values_for_{p}.csv","w")
file.write("N ,FRDM,RMF ,Exp,This work,P_e,R_n,R_p,S_n1,S_n2 \n")
for n in calculated_energys.keys():
    line =str(n)+","+str(frdm_energys.get(n))+","+str(relative_energys.get(n))+","+ str(exp_energys.get(n))+","+ str(calculated_energys[n])+","+str(calculated_pairing_energy[n]) + ","+ str(calculated_Rn[n]) + "," + str(calculated_Rp[n]) + ","+str(calculate_Sn1[n])+","+ str(calculate_Sn2[n])+ "\n"
    file.write(line)

file.close()
"""

# Define a list of markevery cases and color cases to plot
cases = ["FRDM",
         "this work",
         "RMF",
         "exp"]

colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728']

plt.xlabel('nombre des netrons N')
plt.ylabel('$BE/A [MEV]$')
plt.title(title)

methods = [frdm_energys , calculated_energys ,relative_energys , exp_energys ]


for i in range(len(methods)):
    frdm_energys = methods[i]
    maxi = 0
    for n in exp_energys:
        maxi = max(maxi ,abs( frdm_energys[n]/(n+p)-exp_energys[n]/(n+p) ))

    print(maxi)
    x= list(frdm_energys.keys())
    x = np.array(x)
    y = list(frdm_energys.values())
    y = np.array(y) / (x+p)
    plt.plot(x,y, marker='.', label=str(cases[i]))
    plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0.)


plt.grid()
plt.show()
