# Sheet 1, Problem 2
# Write a procedure which determines the interquartile range of a set of numbers. (Base your algorithm on the
# quicksort algorithm and only sort that parts necessary.) Test your program. Plot run time versus data size.

import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt

# Erstellung zufälliger Zahlen

mean = 10
sigma = 3
cnt = 5

data = np.random.normal(loc=mean, scale=sigma, size=cnt)

print(data)
print(len(data))


#--------------------------------------------------------------------------------------------------------------------------------------------
# Funktionen zur Bestimmung des Interquartilsabstands

def quicksort(numbers):
    
    pivot_delete = range(0, len(numbers))

    while True:
        low = []
        high = []
        numbers_m = numbers
        p_pivot = np.random.choice(pivot_delete)
        pivot = numbers[p_pivot]
        print('Pivot, p_pivot =', pivot, p_pivot)
        for i in range(0, len(numbers)):
            if pivot > numbers[i]:
                low = np.append(low, numbers[i])
                print('low =', low)
            elif pivot < numbers[i]:
                high = np.append(high, numbers[i])
                print('high =', high)
            else:
                continue
        low = np.append(low, pivot)
        numbers = np.concatenate((low, high))
        print('Numbers =', numbers)
        pivot_delete = np.delete(pivot_delete, p_pivot)
        print('pivot_delet', pivot_delete)

        print('len(pivot_delete) =', len(pivot_delete))

        #break

        if len(pivot_delete) == 0:
            return numbers


print('sorted data:', quicksort(data))
                


 
#def interquart(numbers):
    