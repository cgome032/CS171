#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Iris information
irisFileString = "/home/carlos/Documents/CS171/Assignment1/IrisData/iris.data"
irisHeaders = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
irisData = np.genfromtxt(irisFileString, delimiter=',', dtype=None, names=irisHeaders)
print(irisData)

plt.hist(irisData['sepal_length'], bins=5)
plt.title("Sepal Length")
plt.xlabel("Length")
plt.ylabel("Frequency")

plt.hist(irisData['sepal_width'], bins=5)
plt.title("Sepal Width")
plt.xlabel("Width")
plt.ylabel("Frequency")


# Wine information
wineFileString = "/home/carlos/Documents/CS171/Assignment1/WineData/wine.data"

#wineData = 

