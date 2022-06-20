import matplotlib.pyplot as plt
import numpy as np

def factorial(x):
    if (x>=1):
        return x*factorial(x-1)
    return 1

x = type(print(3))
print("here")
print(x)
exit(1)
x =np.linspace(0,12,3000)
y = [factorial(i) for i in x]
y =np.array(y)
print(type(y))
plt.plot(x,y)
plt.show()