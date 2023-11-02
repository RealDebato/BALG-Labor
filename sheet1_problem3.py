import numpy as np
import math
import scipy as sp
import cv2
import matplotlib.pyplot as plt


#Kermit = cv2.imread('_Smiley.png', 0)
#cv2.imshow('Orginal Kermit', Kermit)
#Kermit = np.ones((100, 100)).astype('uint8')

Kermit = np.array([[0, 0, 50, 0, 0],
                  [0, 170, 180, 160, 0],
                  [90, 210, 255, 190, 110],
                  [0, 140, 190, 150, 0],
                  [0, 0, 40, 0, 0]])

print('Orginal Kermit\n',Kermit)

def hist_n(img):
    histogram = np.zeros(256)
    
    if img.ndim == 2:
       for i in range(0, img.shape[0]):
              for j in range(0, img.shape[1]):
                     GV = img[i, j]
                     GV = int(GV)                     
                     histogram[GV] = histogram[GV] + 1                     
       histogram_normal = histogram/histogram.sum()
       return histogram_normal
    elif img.ndim == 1:
       for i in range(0, img.shape[0]):
              GV = img[i]
              GV = int(GV)                       # Warum hier nötig aber nicht bei if img.ndim == 2 ???
              histogram[GV] = histogram[GV] + 1              
       histogram_normal = histogram/histogram.sum()
       return histogram_normal

Histn_Kermit = hist_n(Kermit)
max_hist = max(Histn_Kermit)

def Entropy(p):                                         # Argument sind die Wahrscheinlichkeiten als Vektor 
       E = -(p[p > 0] * np.log2(p[p > 0])).sum()
       return E

print(Entropy(Histn_Kermit))


def entropy_filter(img, r):                 # img[col, row]
       
       gv_core1 = []
       gv_col1 = []
       gv_col4 = []
       m_img = img
       
       # Randberech erweitern
       img = np.pad(array=img, pad_width=r, mode='constant', constant_values=0)
       m_img = np.pad(array=m_img, pad_width=r, mode='constant', constant_values=0)
       #print('Img Pad=', img)


       for i in range(r, img.shape[0] - r):             # Laufvariable durch das Bild in y-Richtung (Reihen)
              for j in range(r, img.shape[1] - r):      # Laufvariable durch das Bild in x-Richtung (Spalten)

                     # 1. Filterkern wird vollständig berechnet
                     if j == r and i == r:                     
                            for k in range(-r, r+1):           # k:=row      q:=col   
                                   for q in range(-r, r+1):
                                          gv = m_img[i + k, j + q]
                                          gv_core1 = np.append(gv_core1, gv)

                            print('gv_core1 =', gv_core1)
                            E_core1 = Entropy(hist_n(gv_core1))
                            print('E_core1 =', E_core1)
                            m_pre_E_Core = E_core1
                            img[r, r] = 32 * E_core1
                            gv_core1 = []
                            print('Ende i == r')
                            print('-------------------------------------')
                     else:
                           continue 

                     # 1. Col aus Core1 berechnen zur Subtraktion
                     for n in range(-r, r+1):           
                            gv = m_img[i + n, j - 1]
                            gv_col1 = np.append(gv_col1, gv)
                     m_col1 = gv_col1
                     gv_col1 = []
                     E_col1 = Entropy(hist_n(m_col1))   
                               
                     # 4. Col berechnen zur Addition am nächsten Pixel
                     for s in range(-r, r + 1):                                      
                            gv = m_img[j + s, i + r]
                            gv_col4 = np.append(gv_col4, gv)
                     m_col4 = gv_col4
                     gv_col4 = []
                     print('gv_col4 =', m_col4)
                     print('---------------------------------')
                     E_col4 = Entropy(hist_n(m_col4))

                     m_E_Core = m_pre_E_Core + E_col4 - E_col1
                     print('m_E_Core =', m_E_Core)
                     img[i, j] = 32 * m_E_Core
                     m_pre_E_Core = m_E_Core
                     gv_col4 = []

       return img

                            
'''gw = img[i + k, j + n]
#print('gw(i + k, j + n) =', gw)
gw = int(gw)
p = np.append(p, gw)
#print('p =', p)'''

'''histn_p = hist_n(p)
core = core + Entropy(histn_p)
img[i, j] = core'''

       
Kermit_Entropy_filtered = entropy_filter(Kermit, 1)
print('Filtered Kermit=\n', Kermit_Entropy_filtered)

#cv2.imshow('Entropie-Bild', Kermit_Entropy_filtered)
print('Ende')
#cv2.waitKey(0)
#cv2.destroyAllWindows() 