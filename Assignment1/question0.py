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

    
histogram(setosa['sepal_length'], 10, 1, 1,"Setosa Sepal Length")
histogram(setosa['sepal_width'], 10, 1, 2, "Setosa Sepal Width")
histogram(setosa['petal_length'], 10, 1, 3, "Setosa Petal Length")
histogram(setosa['petal_width'], 10, 1, 4, "Setosa Petal Width")

plt.tight_layout()
plt.show()

histogram(versicolor['sepal_length'], 10, 1, 1,"Versicolor Sepal Length")
histogram(versicolor['sepal_width'], 10, 1, 2, "Versicolor Sepal Width")
histogram(versicolor['petal_length'], 10, 1, 3, "Versicolor Petal Length")
histogram(versicolor['petal_width'], 10, 1, 4, "Versicolor Petal Width")

plt.tight_layout()
plt.show()

histogram(virginica['sepal_length'], 10, 1, 1,"Virginica Sepal Length")
histogram(virginica['sepal_width'], 10, 1, 2, "Virginica Sepal Width")
histogram(virginica['petal_length'], 10, 1, 3, "Virginica Petal Length")
histogram(virginica['petal_width'], 10, 1, 4, "Virginica Petal Width")

plt.tight_layout()
plt.show()





plt.figure(2)
plt.boxplot(irisData['sepal_length'],vert=False)
plt.show()

"""
Question 2 
"""

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

def correlation(x,y):
    return (covariance(x,y)/(stdDeviation(x) * stdDeviation(y)))


def corrMatrix(Dataset):    
    cMatrix = []
    for firstName in Dataset.dtype.names:
        if firstName == 'label':
            continue
        newRow = []
        for secondName in Dataset.dtype.names:
            if secondName != 'label' and firstName != 'label':
                corr = correlation(Dataset[firstName], Dataset[secondName])
                newRow.append(corr)
                print(corr,end=' ')
        print()
        cMatrix.append(newRow)
    return np.array(cMatrix)

testMatrix = corrMatrix(irisData)

plt.figure(1)
plt.imshow(testMatrix,cmap='GnBu')

cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
plt.show

"""
Question 2-2
Scatterplots of the features vs features
"""
scatterIndex = 1
plt.figure(2, figsize=(20,20))
for yname in irisHeaders:
    if yname != 'label':
        for xname in irisHeaders:
            if xname != 'label':
                plt.subplot(4,4, scatterIndex)
                plt.scatter(setosa[xname], setosa[yname],s=50, c='red')
                plt.scatter(versicolor[xname], versicolor[yname],s=50, c = 'green')
                plt.scatter(virginica[xname], virginica[yname],s=50, c='blue')
                scatterIndex+=1
plt.show



"""
Question 3
"""

def distance(x,y,p):
    totalDistance = 0
    for i,j in zip(x,y):
        totalDistance = abs(i-j)**p
    return totalDistance**(1/p)

def createLPmatrix(Dataset,p):
    lpMatrix = []
    for firstName in Dataset.dtype.names:
        if firstName == 'label':
            continue
        newRow = []
        for secondName in Dataset.dtype.names:
            if secondName != 'label' and firstName != 'label':
                lpRow = distance(Dataset[firstName], Dataset[secondName],p)
                newRow.append(lpRow)
                #print(lpRow,end=' ')
        #print()
        lpMatrix.append(newRow)
    return np.array(lpMatrix)

testMatrix = createLPmatrix(irisData,1)
plt.figure(3)
plt.imshow(testMatrix, cmap='Greys')
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
plt.show

            

print("This is the distance function practice run")

print(distance(setosa['sepal_length'],setosa['sepal_width'],1))