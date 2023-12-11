import numpy as np
import math
import scipy
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
import time

#---------------------------------------------------------------------------------------------------
# functions

def plot_image_to_3D(plot_3d):

    plot_3d = np.asarray(plot_3d)

    x = np.linspace(0, plot_3d.shape[1]-1, plot_3d.shape[1])
    y = np.linspace(0, plot_3d.shape[0]-1, plot_3d.shape[0])
    X, Y = np.meshgrid(x , y)

    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, plot_3d, cmap=plt.cm.Blues, linewidth=0)

#---------------------------------------------------------------------------------------------------
# globals

#---------------------------------------------------------------------------------------------------
# images

img = cv2.imread(R'sheet2\FurtherImages\hist.png', 0)

#---------------------------------------------------------------------------------------------------
# main
# morph smooth by reconstruction
#closing by reconstruction
seed_erode = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)
img_blured = ski.morphology.reconstruction(seed_erode, img, method='dilation')
seed_dilate = cv2.dilate(img_blured, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)
img_blured = ski.morphology.reconstruction(seed_dilate, img_blured, method='erosion')
#opening by reconstruction
seed_dilate = cv2.dilate(img_blured, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)
img_blured = ski.morphology.reconstruction(seed_dilate, img_blured, method='erosion')
seed_erode = cv2.erode(img_blured, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1)
img_blured = ski.morphology.reconstruction(seed_erode, img_blured, method='dilation')

img_blured = 255 - img_blured

plt.figure()
plt.imshow(img_blured)
plt.title('Blured Img')

# -----> Bild ist schön glatt


# foreground ermitteln-------------------------------
# Morphological smoothing closing + opening

local_max = ski.feature.peak_local_max(img_blured, min_distance=50)     # Coord. von Maxima Peaks im Bild ermitteln
fg_local_max = np.ones_like(img_blured)
fg_local_max[tuple(local_max.T)] = 255                                  # Peaks in neuem Bild einzeichnen
fg_local_max[fg_local_max!=255] = 0

'''plt.figure()
plt.imshow(fg_local_max)
plt.title('Foreground Local Max')'''

fg_threshold = np.ones_like(img_blured)                                 # Threshold marker ermitteln 
fg_threshold[img_blured>92] = 255
fg_threshold[img_blured<=92] = 0 

fg_threshold = cv2.erode(fg_threshold, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) # Marker verkleinern und glätten
fg_threshold = cv2.medianBlur(fg_threshold.astype(np.uint8), 5)

plt.figure()
plt.imshow(fg_threshold)
plt.title('Foreground Threshold')

# background-------------------------------
# SKIZ erstellen
# Distance map zwischen den Foreground-Markern erstellen
# Auf der Distance Map watershed anwenden, sodass die Bereiche zwischen den Markern gefüllt werden
# Watershedlinien entsprechen dem SKIZ
# Idee: 2. Ableitung müsste Maximas entlang des SKIZ haben. Laplacian auf Distance Map anwenden?

inv_fg = np.zeros_like(fg_threshold)
inv_fg[fg_threshold==0] = 255

'''plt.figure()
plt.imshow(inv_fg)
plt.title('inv fg')'''

dmap_sk = cv2.distanceTransform(inv_fg.astype(np.uint8), cv2.DIST_L2, 3)

plot_image_to_3D(dmap_sk)

#skiz = ski.filters.gaussian(dmap_sk, 10)           # Bild müsste mehr geglättet werden, sonst zu viele Wendepunkte in Distance Map
#skiz = cv2.Laplacian(dmap_sk, ddepth=-1)

labels_fg_threshold = ndimage.label(fg_threshold)[0]        # Nummerierung aller BOLBs im Foreground

'''plt.figure()
plt.imshow(labels_fg_threshold)
plt.title('labels_sk')'''

skiz = ski.segmentation.watershed(dmap_sk, labels_fg_threshold, watershed_line=True)        # Watershed entlang der Distance Map

skiz_lines = np.zeros_like(skiz)        # Es werden nur die Watershed-Linien benötigt
skiz_lines[skiz==0] = 1

'''plt.figure()
plt.imshow(skiz_lines)
plt.title('SKIZ')'''


'''gradient_b = gradient
gradient_b[gradient>4] = 255
gradient_b[gradient<=4] = 0
gradient_for_skeletonize = cv2.dilate(gradient, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))'''


marker_fg_local_max = ndimage.label(fg_local_max)[0]            
marker_fg_threshold = ndimage.label(fg_threshold)[0]

'''plt.figure()
plt.imshow(marker_fg_local_max)
plt.title('Marker FG-BG local max')

plt.figure()
plt.imshow(marker_fg_threshold)
plt.title('Marker FG-BG threshold')'''

gradient = cv2.morphologyEx(img_blured, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))   # Gradient für Watershed

watershed_fg_local_max = ski.segmentation.watershed(gradient, marker_fg_local_max, watershed_line=True)     # Watershed ohne Background
watershed_fg_threshold = ski.segmentation.watershed(gradient, marker_fg_threshold, watershed_line=True)



marker_fg_bg_local_max = marker_fg_local_max + 1                        # Alle Marker werden um 1 erhöht, sodass das Markerlabel 0 frei/unbenutzt ist
flood_area_local_max = fg_local_max.astype(np.uint8) + skiz_lines       # Das Flutgebiet soll zwischen FG und BG liegen. Addition von FG und BG, dann ist alles ==0 das Flutgebiet

'''plt.figure()
plt.imshow(flood_area_local_max)
plt.title('Flood area local max')'''

marker_fg_bg_local_max[flood_area_local_max == 0] = 0


marker_fg_bg_threshold = marker_fg_threshold + 1
flood_area_threshold = fg_threshold.astype(np.uint8) + skiz_lines           # eigentliche flood area bei [flood_area==0]

'''plt.figure()
plt.imshow(flood_area_threshold)
plt.title('Flood area threshold')'''

marker_fg_bg_threshold[flood_area_threshold == 0] = 0

'''plt.figure()
plt.imshow(marker_fg_bg_local_max)
plt.title('Marker FG-BG local max')

plt.figure()
plt.imshow(marker_fg_bg_threshold)
plt.title('Marker FG-BG threshold')'''

watershed_fg_bg_local_max = ski.segmentation.watershed(gradient, marker_fg_bg_local_max, watershed_line=True)       # watershed mit FG+BG
watershed_fg_bg_threshold = ski.segmentation.watershed(gradient, marker_fg_bg_threshold, watershed_line=True)

#---------------------------------------------------------------------------------------------------
# output

plt.figure()
plt.imshow(watershed_fg_local_max)
plt.title('Watershed FG local max')

plt.figure()
plt.imshow(watershed_fg_threshold)
plt.title('Watershed FG threshold')

plt.figure()
plt.imshow(watershed_fg_bg_local_max)
plt.title('Watershed FG-BG local max')

plt.figure()
plt.imshow(watershed_fg_bg_threshold)
plt.title('Watershed FG-BG threshold')

'''plt.figure()
plt.imshow(img)
plt.title('Orginal')'''

'''plt.figure()
plt.imshow(gradient)
plt.title('Gradient')'''

'''plt.figure()
plt.imshow(gradient_smoothed)
plt.title('Smoothed gradient')'''

'''plt.figure()
plt.imshow(watershed_on_gradient)
plt.title('Watershed on gradient map')

plot_image_to_3D(gradient)'''

#---------------------------------------------------------------------------------------------------
# main-end

plt.show()
print('La fin')
cv2.waitKey(0)