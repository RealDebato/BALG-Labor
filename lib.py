import numpy as np
import time

def getEntropy(array):
    array = np.array(array).flatten()
    hist, _ = np.histogram(array, 256, [0,256], False)
    hist = np.array(hist, dtype=np.float64)
    entropy = -np.sum(hist[hist != 0]/sum(hist) * np.log2(hist[hist != 0]/sum(hist)))
    return entropy

def getKernelCirc(img, x, y, r=3):
    img = np.array(img)
    X, Y = np.meshgrid(np.linspace(0, img.shape[1]-1, img.shape[1]), np.linspace(0, img.shape[0]-1, img.shape[0]))
    kernel = img[ np.sqrt((X - x)**2 + (Y - y)**2) <= r ]
    return kernel

def getEntropyOptimized(img, r=3):
    img = np.array(img)
    entropy = img*0.0
    X, Y = np.meshgrid(np.linspace(0, img.shape[1]-1, img.shape[1]), np.linspace(0, img.shape[0]-1, img.shape[0]))
    for y in range(img.shape[0]):
        reloadHist = True
        for x in range(img.shape[1]):
            if reloadHist:
                oldKernel = np.sqrt((X - x)**2 + (Y - y)**2) <= r
                elements = img[oldKernel]
                hist, _ = np.histogram(elements, 256, [0,256], False)
                relativeFrequency = (hist[hist != 0]/np.sum(hist))
                entropy[y,x] = -np.sum(relativeFrequency * np.log2(relativeFrequency))
                reloadHist = False
            else:
                newKernel = np.sqrt((X - (x))**2 + (Y - y)**2) <= r
                leftKernel = np.logical_and(oldKernel == True, newKernel == False)
                rightKernel = np.logical_and(oldKernel == False, newKernel == True)
                oldElements = img[leftKernel]
                newElements = img[rightKernel]
                
                # Das bringt keine Optimierung, da die Berechnung eines kompletten Histogramms zu viel kostet
                # oldHist, _ = np.histogram(oldElements, 256, [0,256], False)
                # newHist, _ = np.histogram(newElements, 256, [0,256], False)
                # hist = hist - oldHist + newHist

                for i in oldElements:
                    hist[i] -= 1
                for i in newElements:
                    hist[i] += 1
                
                oldKernel = newKernel
                relativeFrequency = (hist[hist != 0]/np.sum(hist))
                entropy[y,x] = -np.sum(relativeFrequency * np.log2(relativeFrequency))
    return entropy

def getEntropyFurtherOptimized(img, r=3):
    t0 = time.time()
    maxPossibleElements = (2*r)**2
    pldp = np.zeros([maxPossibleElements+1,maxPossibleElements+1])
    pldp[1,1] = 0
    for i in range(1,maxPossibleElements+1):
        for j in range(1,i):
            relativeFrequency = j/i
            pldp[j,i] = relativeFrequency * np.log2(relativeFrequency)
    t1 = time.time()
    print("Time for LUT calc: " + str(t1-t0))
    img = np.array(img)
    entropy = img*0.0
    X, Y = np.meshgrid(np.linspace(0, img.shape[1]-1, img.shape[1]), np.linspace(0, img.shape[0]-1, img.shape[0]))
    for y in range(img.shape[0]):
        reloadHist = True
        
        for x in range(img.shape[1]):
            if reloadHist:
                oldKernel = np.sqrt((X - x)**2 + (Y - y)**2) <= r
                elements = img[oldKernel]
                hist, _ = np.histogram(elements, 256, [0,256], False)
                histElements = hist[hist != 0]
                entropy[y,x] = -np.sum(pldp[tuple(histElements),tuple(histElements*0+np.sum(hist))])
                reloadHist = False
            else:
                newKernel = np.sqrt((X - (x))**2 + (Y - y)**2) <= r
                leftKernel = np.logical_and(oldKernel == True, newKernel == False)
                rightKernel = np.logical_and(oldKernel == False, newKernel == True)
                oldElements = img[leftKernel]
                newElements = img[rightKernel]

                for i in oldElements:
                    hist[i] -= 1
                for i in newElements:
                    hist[i] += 1
                
                oldKernel = newKernel
                histElements = hist[hist != 0]
                entropy[y,x] = -np.sum(pldp[tuple(histElements),tuple(histElements*0+np.sum(hist))])
    return [entropy, t1-t0]