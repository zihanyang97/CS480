import numpy as np
from numpy import *
import csv
import math
import operator
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


# function to import train data
def importData(file):
    data = []
    csvFile = open(file, "r")
    csv_reader = csv.reader(csvFile, delimiter=',')
    
    for row in csv_reader:
        tmp = []
        for col in row:
            tmp.append(float(col))
        data.append(tmp)
        
    return data

def gaussianProcess(d, xTrain, xValidation, yTrain, yValidation):
    MSE = 0
    # get I 
    I = np.zeros((len(xTrain),len(xTrain)))
    for i in range(len(I)):
        I[i][i] = 1
    # get gram matrix
    K = np.zeros((len(xTrain),len(xTrain)))
    for i in range(len(xTrain)):
        for j in range(len(xTrain)):
            K[i,j] = (dot(xTrain[i], xTrain[j]) + 1)**d
            
    k = np.zeros((len(xValidation), len(xTrain)))
    for i in range(len(xValidation)):
        for j in range(len(xTrain)):
            k[i,j] = (dot(xValidation[i], xTrain[j]) + 1)**d
    # compute
    K = K + I
    K = np.matrix(K)
    KI = K.getI()
    #print(K.shape)
    #print(K)
    pred = dot(k, dot(KI, yTrain))
    #print(pred)
    #print(yValidation)
    for i in range(len(xValidation)):
        MSE += float(pred[i] - yValidation[i])**2
    return MSE / len(xValidation)


# import train data
trainInput = []
trainInput += importData("trainInput1.csv")
trainInput += importData("trainInput2.csv")
trainInput += importData("trainInput3.csv")
trainInput += importData("trainInput4.csv")
trainInput += importData("trainInput5.csv")
trainInput += importData("trainInput6.csv")
trainInput += importData("trainInput7.csv")
trainInput += importData("trainInput8.csv")
trainInput += importData("trainInput9.csv")
trainInput += importData("trainInput10.csv")

#print(trainInput)
#print(len(trainInput))

# import train labels
trainTarget = []
trainTarget += importData("trainTarget1.csv")
trainTarget += importData("trainTarget2.csv")
trainTarget += importData("trainTarget3.csv")
trainTarget += importData("trainTarget4.csv")
trainTarget += importData("trainTarget5.csv")
trainTarget += importData("trainTarget6.csv")
trainTarget += importData("trainTarget7.csv")
trainTarget += importData("trainTarget8.csv")
trainTarget += importData("trainTarget9.csv")
trainTarget += importData("trainTarget10.csv")

#print(trainTarget)
#print(len(trainTarget))


# import test data
testInput = importData("testInput.csv")

# import test labels
testTarget = importData("testTarget.csv")

# split dataset for 10-cross validation

# 1
xValidation1 = trainInput[0:20]
yValidation1 = trainTarget[0:20]
xTrain1 = trainInput[20:200]
yTrain1 = trainTarget[20:200]

#print xValidation1

# 2 
xValidation2 = trainInput[20:40]
yValidation2 = trainTarget[20:40]
xTrain2 = trainInput[0:20] + trainInput[40:200]
yTrain2 = trainTarget[0:20] + trainTarget[40:200]

# 3
xValidation3 = trainInput[40:60]
yValidation3 = trainTarget[40:60]
xTrain3 = trainInput[0:40] + trainInput[60:200]
yTrain3 = trainTarget[0:40] + trainTarget[60:200]

# 4
xValidation4 = trainInput[60:80]
yValidation4 = trainTarget[60:80]
xTrain4 = trainInput[0:60] + trainInput[80:200]
yTrain4 = trainTarget[0:60] + trainTarget[80:200]

# 5
xValidation5 = trainInput[80:100]
yValidation5 = trainTarget[80:100]
xTrain5 = trainInput[0:80] + trainInput[100:200]
yTrain5 = trainTarget[0:80] + trainTarget[100:200]

# 6
xValidation6 = trainInput[100:120]
yValidation6 = trainTarget[100:120]
xTrain6 = trainInput[0:100] + trainInput[120:200]
yTrain6 = trainTarget[0:100] + trainTarget[120:200]

# 7
xValidation7 = trainInput[120:140]
yValidation7 = trainTarget[120:140]
xTrain7 = trainInput[0:120] + trainInput[140:200]
yTrain7 = trainTarget[0:120] + trainTarget[140:200]

# 8
xValidation8 = trainInput[140:160]
yValidation8 = trainTarget[140:160]
xTrain8 = trainInput[0:140] + trainInput[160:200]
yTrain8 = trainTarget[0:140] + trainTarget[160:200]

# 9
xValidation9 = trainInput[160:180]
yValidation9 = trainTarget[160:180]
xTrain9 = trainInput[0:160] + trainInput[180:200]
yTrain9 = trainTarget[0:160] + trainTarget[180:200]

# 10
xValidation10 = trainInput[180:200]
yValidation10 = trainTarget[180:200]
xTrain10 = trainInput[0:180]
yTrain10 = trainTarget[0:180]

degreeRange = np.arange(1, 5)
minMSE = float('inf')
score_list = []
bestdegree = 0

for i in degreeRange:
    MSE1 = gaussianProcess(i, xTrain1, xValidation1, yTrain1, yValidation1)
    MSE2 = gaussianProcess(i, xTrain2, xValidation2, yTrain2, yValidation2)
    MSE3 = gaussianProcess(i, xTrain3, xValidation3, yTrain3, yValidation3)
    MSE4 = gaussianProcess(i, xTrain4, xValidation4, yTrain4, yValidation4)
    MSE5 = gaussianProcess(i, xTrain5, xValidation5, yTrain5, yValidation5)
    MSE6 = gaussianProcess(i, xTrain6, xValidation6, yTrain6, yValidation6)
    MSE7 = gaussianProcess(i, xTrain7, xValidation7, yTrain7, yValidation7)
    MSE8 = gaussianProcess(i, xTrain8, xValidation8, yTrain8, yValidation8)
    MSE9 = gaussianProcess(i, xTrain9, xValidation9, yTrain9, yValidation9)
    MSE10 = gaussianProcess(i, xTrain10, xValidation10, yTrain10, yValidation10)
    avgMSE = (MSE1 + MSE2 + MSE3 + MSE4 + MSE5 + MSE6 + MSE7 + MSE8 + MSE9 + MSE10) / 10
    if avgMSE < minMSE:
        minMSE = avgMSE
        bestdegree = i
    print("when degree = " + str(i))
    print("MSE for Gaussian Process with polynomial is " + str(avgMSE))
    score_list.append(avgMSE)
    

print("best degree is " + str(bestdegree))

result = gaussianProcess(bestdegree, trainInput, testInput, trainTarget, testTarget)

print("MSE using best degree " + str(bestdegree) + " for test set is " + str(result))

plt.plot(degreeRange, score_list)
plt.xlabel("value of degree for gaussian process with polynomial kernel")
plt.ylabel("MSE")
plt.show()