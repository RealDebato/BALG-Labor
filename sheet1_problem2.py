# Sheet 1, Problem 2
# Write a procedure which determines the interquartile range of a set of numbers. (Base your algorithm on the
# quicksort algorithm and only sort that parts necessary.) Test your program. Plot run time versus data size.

import numpy as np
import math
import scipy as sp
import time


t_start = time.time()

# Erstellung zufälliger Zahlen

mean = 10
sigma = 2
cnt = 1000000

data = np.random.normal(loc=mean, scale=sigma, size=cnt)
#data = [8, 7, 2, 5, 4, 6, 1, 10, 11]
#print(data)

#--------------------------------------------------------------------------------------------------------------------------------------------
# Funktionen zur Bestimmung des Interquartilsabstands

def partition(A, low, high):
    pivot = A[high]
    i = low - 1

    for j in range(low, high):
        if A[j] <= pivot:
            i = i + 1
            A[i], A[j] = A[j], A[i]
        
    A[i + 1], A[high] = A[high], A[i + 1]
    #print('i =', i)
    #print('j =', j)
    #print('Numbers =', A)
    return i + 1

def quickSort(A, low, high):
    if low < high:
        pi = partition(A, low, high)
        quickSort(A, low, pi - 1)
        quickSort(A, pi + 1, high)
        return A

sorted = quickSort(data, 0, len(data) - 1)
#print(sorted)

'''def quicksort(numbers, left, right):
    if left < right:
        partition_pos = partition(numbers, left, right)
        print('Pos P=', partition_pos)
        if left < numbers[partition_pos - 1]:
            quicksort(numbers, left, partition_pos - 1)
        if numbers[partition_pos + 1] < right:
            quicksort(numbers, partition_pos + 1, right)
    return numbers
    

def partition(numbers, left, right):
    i = left
    j = right - 1
    p_pivot = np.random.choice(range(left, right))
    pivot = numbers[p_pivot]
    numbers[p_pivot], numbers[right] = numbers[right], numbers[p_pivot]
    
    while i < j:
        while numbers[i] < pivot and i < right:
            i = i + 1
        while numbers[j] >= pivot and left < j:
            j = j - 1
        if numbers[i] > numbers[j]:
            numbers[i], numbers[j] = numbers[j], numbers[i]
    
    if pivot < numbers[i]:
        numbers[i], numbers[p_pivot] = numbers[p_pivot], numbers[i]
        

    return i 


sort = quicksort((data), 0, len(data) - 1)

print(sort)

    



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
#print('sorted data:', sorted)'''

t_end_1 = time.time()

print('Laufzeit Quicksort =', t_end_1 - t_start,'s')

#------------------------------------------------------------------------------------------------------------
# Berechnung IQR
                
def iqr(numbers):
    # Median x1 des 1. Quartiels berechnen
    a = round(cnt/4)
    b = round(cnt/4 * 3)

    if (cnt % 4) == 0:
        #print('Test bei cnt % 4 == 0')
        x1 = 0.5 * (numbers[a] + numbers[a + 1])
    else:
        x1 = numbers[a + 1]
    
    # Median x3 des 3. Quartiels berechnen
    if cnt % 4  == 0:  
        x3 = 0.5 * (numbers[b] + numbers[b + 1])
    else:
        x3 = numbers[b + 1]
    #print('x1 , x3 =', x1, x3)
    IQR = x3 - x1
    return IQR


print('IQR =', iqr(sorted))
   