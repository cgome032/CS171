#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import math

import urllib.request

"""
Question 0
"""

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

raw_data = urllib.request.urlopen(url)


irisHeaders = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label']
irisData = np.genfromtxt(raw_data, delimiter=',', dtype=None, names=irisHeaders)

setosa = irisData[0:50]
versicolor = irisData[50:100]
virginica = irisData[100:150]

#print(setosa)
#print(versicolor)
#print(virginica)


#print(irisData['sepal_length'])
#print(irisData)   

# Wine information
wineFileString = "/home/carlos/Documents/CS171/Assignment1/WineData/wine.data"


"""
Question 1

Implemented a histogram function


"""

def histogram(dataArray, binSize, figure, subIndex, title):
    arrayMin = np.min(dataArray)
    arrayMax = np.max(dataArray)
    print("Array minimum: ", arrayMin)
#    print("Array maximum: ", arrayMax)
    width = (arrayMax - arrayMin)/binSize
#    print("Bar width: ", width)
    
    # Index to keep track of bar data
    lowerBound = arrayMin
    graphBars = []
    indicies = []
    
    
    for bar in range(binSize):
        if bar != binSize-1:
            # Zero bar - binSize-1 bar is inclusive on left side and exclusive on right side
            frequency = ((lowerBound <= dataArray) & (dataArray < lowerBound+width)).sum()
            #print("Frequency: ", frequency)
            graphBars.append(frequency)
        else:
            # Middle bars are inclusive on right side and exclusive on left side
            frequency = ((lowerBound < dataArray) & (dataArray <= lowerBound+width)).sum()
            #print("Frequency:", frequency)
            graphBars.append(frequency)
        indicies.append(lowerBound)
        lowerBound+=width
           
            
    plt.figure(figure)
    plt.subplot(2,2,subIndex)
    plt.bar(indicies, graphBars, width, align='edge')
    plt.title(title)
    plt.xticks(np.arange(arrayMin, arrayMax,width))
    
histogram(irisData['sepal_length'], 5, 1, 1,"Sepal Length")
histogram(irisData['sepal_width'], 5, 1, 2, "Sepal Width")
histogram(irisData['petal_length'], 5, 1, 3, "Petal Length")
histogram(irisData['petal_width'], 5, 1, 4, "Petal Width")

plt.tight_layout()
plt.show()

plt.figure(2)
plt.boxplot(irisData['sepal_length'],vert=False)
plt.show()

"""
Question 2 
"""

def correlation(x,y):
    return 0

def covariance(x,y):
    xMean = np.mean(x)
    yMean = np.mean(y)
    numerator = 0
    for i,j in zip(x,y):
        numerator += (i*j)
    return (numerator/len(x)) - xMean * yMean

def stdDeviation(a):
    aMean = np.mean(a)
    sum = 0
    for i in a:
        sum += i**2
    return math.sqrt((sum / len(a)) - (aMean ** 2))

testA = np.array([30,36,47,50,52,52,56,60,63,70,70,110])
print("Standard Deviation: ", stdDeviation(testA))
            
testX = np.array([6,5,4,3,2])
testY = np.array([20,10,14,5,5])

print("Covariance: ",covariance(testX,testY))
