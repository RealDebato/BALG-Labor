# Sheet 1, Problem 2
# Write a procedure which determines the interquartile range of a set of numbers. (Base your algorithm on the
# quicksort algorithm and only sort that parts necessary.) Test your program. Plot run time versus data size.

import numpy as np
import math
import scipy as sp

# Erstellung zufälliger Zahlen


def rand(a, b, i):                  # a = Kleinstes Element; b = größtes Element; i = Anzahl der Zufallszahlen
    Random = []
    for i in range(i):
        number = np.random.choice(np.arange(a, b))
        Random = np.add(Random, number)
    return Random

print(rand(0, 10, 100))


    
