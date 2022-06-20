import numpy as np 
from  math import *

linges = 22
columns = linges

centre = linges // 2

a = np.zeros((linges, columns))   # Create an array of all zeros

for x in range(linges):
    for y in range(columns):
        if sqrt(pow(x-centre+1, 2) + pow(y-centre+1, 2)) <= 5:
            a[x][y] = 1

a[centre-1][centre-1] = 9
print (centre)
print (a)
