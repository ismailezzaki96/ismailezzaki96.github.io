import numpy as np
import matplotlib as plt
vector = (1,3,4,5,6)
results = ()
for n in range(0,len(vector)):
    somme = 0
    for m in range(0,len(vector)):
        somme += pow((vector(n)-vector(n+m))/2,2)
    results.append(-somme)

plt.plot(results)