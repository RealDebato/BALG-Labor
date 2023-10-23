# -*- coding: utf-8 -*-
"""

Author: Sandau
Minor Updates: Weinmann


Input: Binärbild, Nichtnull-Wert gehört zum Vordergrund

Jede Zeile des (gefüllten) Vordergrunds wird als Interval gespeichert.  

Output: np.Array mit drei Spalten 
Spalte 0 gibt den Medianindex (mj) bzgl rechtecksarray (links) an
Spalte 1 gibt die Breite in jeder zeile an unter Ignorieren von Löchern
(d.h. falls doch keine v_umgebung wird sie dazu gemacht)
Spalte 2 enthält den Index der ersten 1 in jeder zeile bzgl Rechteckarray

"""
import numpy as np

def vstru2mw(vstru) :
    
    dims = vstru.ndim
    print('dims =', dims)
    sizs = vstru.shape
    print('sizs =', sizs)
    if dims == 1:
       indj,  = np.where(vstru > 0)
       print ('indj =', indj)
       count = len(indj)
       print('count =', count)
       if count > 0 :
            wj =  indj[-1] - indj[0] + 1
            print('wj =', wj)
            mj = (indj[-1] + indj[0] + 1)/2
            print('mj =', mj)
            print('indj[0] =', indj[0])
            mw = np.array([mj,wj,indj[0]])
            print('mw =', mw)
       else :
           mw =  np.array([0,0,0])
           
    if dims == 2:
        mw = np.empty([sizs[0],3])
        for j in range(sizs[0]):            
            indj,= np.where(vstru[j,:] >0)
            count = len(indj)
            print('len =', count)
            if count > 0:
                wj =  indj[-1] - indj[0] + 1
                mj = (indj[-1] + indj[0] + 1)/2  #Ist doch die Länge aber warum dann licht len()?
                mw[j,:] = [mj,wj,indj[0]]
            else:
                mw[j,:] =  [0,0,0]
    return mw
  

#"""
#==========  test  ===========

#vstru = np.array([0,0,1,2,3,2,1,0,0])
#vstru = np.array([[0,0,1,2,3,2,1,0,0],[0,0,0,1,0,0,0,0,0],[0,1,1,2,1,1,1,0,0]])
vstru = np.array([[0,0,1,2,0,2,1,0,0],[0,0,0,1,0,0,0,0,0],[0,1,1,2,1,1,1,1,0]])
    
mw = vstru2mw(vstru)
print(mw)
print(vstru)
#end
#"""

 