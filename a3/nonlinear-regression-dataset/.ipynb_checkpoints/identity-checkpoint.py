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

def gaussianProcess(xTrain, xValidation, yTrain, yValidation):
    MSE = 0
    # get I 
    I = np.zeros((len(xTrain),len(xTrain)))
    for i in range(len(I)):
        I[i][i] = 1
    
    # get gram matrix
    K = np.zeros((len(xTrain),len(xTrain)))
    for i in range(len(xTrain)):
        for j in range(len(xTrain)):
            K[i,j] = dot(xTrain[i], xTrain[j])
            
    k = np.zeros((len(xValidation), len(xTrain)))
    for i in range(len(xValidation)):
        for j in range(len(xTrain)):
            k[i,j] = dot(xValidation[i], xTrain[j])
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

MSE = gaussianProcess(trainInput, testInput, trainTarget, testTarget)
print("MSE for Gaussian Process with identity kernel is " + str(MSE))