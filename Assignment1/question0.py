#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Iris information
irisFileString = "/home/carlos/Documents/CS171/Assignment1/IrisData/iris.data"

irisHeaders = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']

irisData = np.genfromtxt(irisFileString, delimiter=',', dtype=None, names=irisHeaders)
print(irisData)   

# Wine information
wineFileString = "/home/carlos/Documents/CS171/Assignment1/WineData/wine.data"

#wineData = 
def makeHistograms(binSize, DataFrame, figure):
    index = 1
    for name in DataFrame.dtype.names:
        plt.figure(figure)
        if (index <= 4):
            plt.subplot(2,2,index)
            plt.tight_layout()
            plt.hist(DataFrame[name], bins=binSize)
        #plt.title("Histogram title")
            plt.xlabel(name)
            plt.ylabel("Frequency")
        index += 1
        
print("Bin size is 5")
makeHistograms(10,irisData,1)

print("Bin size is 10")
makeHistograms(10,irisData,2)
#
print("Bin size is 50")
makeHistograms(50, irisData,3)

print("Bin size is 100")
makeHistograms(100, irisData,4)