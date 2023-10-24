# Sheet 1, Problem 2
# Write a procedure which determines the interquartile range of a set of numbers. (Base your algorithm on the
# quicksort algorithm and only sort that parts necessary.) Test your program. Plot run time versus data size.

import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt

# Erstellung zufälliger Zahlen


def rand_n(a, b, i):                  # a = Kleinstes Element; b = größtes Element -1 ; i = Anzahl der Zufallszahlen
    Random = []
    j = 0
    for j in range(i):
        number = np.random.choice(np.arange(a, b + 1))
        Random = np.append(Random, number)
    return Random

print(rand_n(0, 10, 100))

'''x = np.linspace(0,50,50)

def gaussian_dist(x , mean , sig):
    density = (np.pi*sig) * np.exp(-0.5*((x-mean)/sig)**2)
    return density


Ran_Num = gaussian_dist(x, 25, 15)
print(Ran_Num)


plt.plot(x, Ran_Num , color = 'red')
plt.xlabel('Data points')
plt.ylabel('Probability Density')'''