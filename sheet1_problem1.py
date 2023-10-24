# Sheet 1, Problem 1
# Write a procedure which determines the minimum of a convex function f in the interval [a,b] using “trisection
# of the interval.” The file containing the description of the function f is called “convFun.” Further use the
# function

import numpy as np
import scipy as sp
import math

#print('Alpha =')
#alpha = input()

#print('Beta =')
#beta = input()

#alpha = 1
#beta = 1


""" def f(var, alpha, beta):
    fy = math.exp(-alpha*var) + pow(var, beta)
    return fy



def convFun(f, a, b, eps):               # f: convexe Funktion, a: untere Intervallgrenze, b: obere Intervallgrenze, eps: gewünschte max. Genauigkeit
    
    e = b - a
    while e > eps:
        if f(a, 1, 1) > f(b, 1, 1):
            a = a - eps
        else:
            b = b - eps
    return fu(a + b/2)

print(convFun(f))


#print(convFun(f, 0, 1, 0.001)) """

def convFun(a, b, eps, alpha, beta):     # Intervall [a,b], Toleranz
    e = b - a
    var = 1
    #math.exp(-alpha*var) + pow(var, beta)
    while e > eps:
        # Grenzen der Trisektion bestimmen
        s = (b - a)/3
        x0 = a
        x1 = a + s
        x2 = b - s
        x3 = b

        Seg1 = math.exp(-alpha*x1) + pow(x1, beta) - math.exp(-alpha*x0) + pow(x0, beta)
        Seg2 = math.exp(-alpha*x2) + pow(x2, beta) - math.exp(-alpha*x1) + pow(x1, beta)
        Seg3 = math.exp(-alpha*x3) + pow(x3, beta) - math.exp(-alpha*x2) + pow(x2, beta)

        print('Seg1 =', Seg1)
        print('Seg2 =', Seg2)
        print('Seg3 =', Seg3)

        if Seg1 > Seg2 < Seg3:
            a = x1
            b = x2
        elif Seg1 < Seg2 and Seg1 < Seg3:
            a = x0
            b = x1
        else:
            a = x2
            b = x3

        e = b - a

        print('e =', e)

    return a + e/2


print('Min =', convFun(0, 1, 0.01, 2, 1))




