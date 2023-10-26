# Sheet 1, Problem 2
# Write a procedure which determines the interquartile range of a set of numbers. (Base your algorithm on the
# quicksort algorithm and only sort that parts necessary.) Test your program. Plot run time versus data size.

import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
import time


t_start = time.time()

# Erstellung zufälliger Zahlen

mean = 10
sigma = 2
cnt = 10

data = np.random.normal(loc=mean, scale=sigma, size=cnt)

#print(data)
#print(len(data))


#--------------------------------------------------------------------------------------------------------------------------------------------
# Funktionen zur Bestimmung des Interquartilsabstands

def quicksort(numbers):
    
    pivot_delete = range(0, len(numbers))
    m_numbers = numbers

    while True:
        low = []
        high = []
        p_pivot = np.random.choice(pivot_delete)
        pivot = m_numbers[p_pivot]
        #print('Pivot, p_pivot =', pivot, p_pivot)
        for i in range(0, len(numbers)):
            if pivot > numbers[i]:
                low = np.append(low, numbers[i])
                #print('low =', low)
            elif pivot < numbers[i]:
                high = np.append(high, numbers[i])
                #print('high =', high)
            else:
                continue
        low = np.append(low, pivot)
        numbers = np.concatenate((low, high))
        #print('Numbers =', numbers)
        p_to_delete = np.where(pivot_delete == p_pivot)
        #print('p_to_delete', p_to_delete)
        pivot_delete = np.delete(pivot_delete, p_to_delete)
        #print('pivot_delet', pivot_delete)

        #print('len(pivot_delete) =', len(pivot_delete))

        #break

        if len(pivot_delete) == 0:
            return numbers

sorted = quicksort(data)
print('sorted data:', sorted)

t_end_1 = time.time()

print('Laufzeit Quicksort =', t_end_1 - t_start,'s')
                
def iqr(numbers):
    # Median x1 des 1. Quartiels berechnen
    if cnt % 4 == 0:
        x1 = 0,5 * (numbers[cnt/4] + numbers[cnt/4 + 1])
    else:
        x1 = numbers[int(cnt/4 + 1)]
    
    # Median x3 des 3. Quartiels berechnen
    if (cnt % 4) * 3 == 0:
        x3 = 0,5 * (numbers[(cnt/4) * 3] + numbers[(cnt/4) * 3 + 1])
    else:
        x3 = numbers[int((cnt/4) * 3 + 1)]
    print('x1 , x3 =', x1, x3)
    
    return x3 - x1


print('IQR =', iqr(sorted))



 
#def interquart(numbers):
    