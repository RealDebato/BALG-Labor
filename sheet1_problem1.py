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

alpha = 5
beta = 1


def f(var, alpha, beta):
    fy = math.exp(-alpha*var) + pow(var, beta)
    return fy

fu = f(1, alpha, beta)

print(fu)


#print(convFun(f, 0, 1, 0.001))

def convFun(a, b, eps):     # Intervall [a,b], Toleranz eps
    
    
    #s_start = (b - a)/3
    #x0_start = a
    #x1_start = a + s_start
    #x2_start = b - s_start
    #x3_start = b

    # Abbruchbedingung: Erst wenn in allen drei Segmenten die Differenz der benachbarten Stüzpunkten in y geringer wie die Toleranz ist, sind wir nah genug am Minimum.  
    #e1 = abs(f(x0, 1, 1) - f(x1, 1, 1))
    #e2 = abs(f(x1, 1, 1) - f(x2, 1, 1))
    #e3 = abs(f(x2, 1, 1) - f(x3, 1, 1))


    while True:
        # Grenzen der Trisektion bestimmen
        s = (b - a)/3
        x0 = a
        x1 = a + s
        x2 = b - s
        x3 = b

        print('X[]', x0, x1, x2, x3)

        #Steigungen zwischen den Segmentgrenzen berechen
        m_seg1 = (f(x1, alpha, beta) - f(x0, alpha, beta))/(x1 - x0)
        m_seg2 = (f(x2, alpha, beta) - f(x1, alpha, beta))/(x2 - x1)
        m_seg3 = (f(x3, alpha, beta) - f(x2, alpha, beta))/(x3 - x2)

        #print('m Seg', m_seg1, m_seg2, m_seg3),
        M_Seg = [m_seg1, m_seg2, m_seg3]
        print('M Seg =', M_Seg)
        M_seg_direction = np.sign(M_Seg)
        print('M Seg Direction =', M_seg_direction)

        # Entsprechend der Steigungen eines der äußeren Segmente löschen
        if np.array_equal(M_seg_direction, [-1,1,1]):
            a = x0
            b = x2
        elif np.array_equal(M_seg_direction, [-1,-1,1]):
            a = x1
            b = x3
        else:
            print('Im Interval wurde keine Minimum gefunden')
            break

        e1 = abs(f(x0, 1, 1) - f(x1, 1, 1))
        e2 = abs(f(x1, 1, 1) - f(x2, 1, 1))
        e3 = abs(f(x2, 1, 1) - f(x3, 1, 1))

        e = (b - a)/2

        if (e1 < eps) & (e2 < eps) & (e3 < eps):
            break
        
        
        
    fx_min = a + e/2
    fy_min = f(a + e/2, 1, 1)
    return fx_min, fy_min


print('Min(x, y) =', convFun(0, 1, 0.01))




