import numpy as np
from numpy import *
import csv
import math
import operator
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
# for testing
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import linear_model

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

def linearRegression(xTrain, xValidation, yTrain, yValidation):
    MSE = 0
    # get I 
    I = np.zeros((len(xTrain[0]),len(xTrain[0])))
    for i in range(len(I)):
        I[i][i] = 1
    # get X
    X = np.matrix(xTrain)
    X = X.getT()
    XT = X.getT()
    A = dot(X, XT) + I
    AI = A.getI()
    # get y
    Y = np.matrix(yTrain)
    W = dot(dot(AI, X), Y)
    
    for i in range(len(xValidation)):
        tmp = np.matrix(xValidation[i])
        tmp = tmp.getT()        
        pred = float(dot(W.getT(), tmp))
        #print("predict = " + str(pred))
        #print("target = " + str(float(yValidation[i][0])))
        MSE += (pred - float(yValidation[i][0]))**2
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
MSElist = []
bestDegree = 0

for i in degreeRange:
    poly = PolynomialFeatures(degree=i)
    
# modify dataset
#1
    modifiedxTrain1 = poly.fit_transform(xTrain1)
    modifiedxValidation1 = poly.fit_transform(xValidation1)
#2
    modifiedxTrain2 = poly.fit_transform(xTrain2)
    modifiedxValidation2 = poly.fit_transform(xValidation2)
#3
    modifiedxTrain3 = poly.fit_transform(xTrain3)
    modifiedxValidation3 = poly.fit_transform(xValidation3)
#4
    modifiedxTrain4 = poly.fit_transform(xTrain4)
    modifiedxValidation4 = poly.fit_transform(xValidation4)
#5
    modifiedxTrain5 = poly.fit_transform(xTrain5)
    modifiedxValidation5 = poly.fit_transform(xValidation5)
#6
    modifiedxTrain6 = poly.fit_transform(xTrain6)
    modifiedxValidation6 = poly.fit_transform(xValidation6)
#7
    modifiedxTrain7 = poly.fit_transform(xTrain7)
    modifiedxValidation7 = poly.fit_transform(xValidation7)
#8
    modifiedxTrain8 = poly.fit_transform(xTrain8)
    modifiedxValidation8 = poly.fit_transform(xValidation8)
#9
    modifiedxTrain9 = poly.fit_transform(xTrain9)
    modifiedxValidation9 = poly.fit_transform(xValidation9)
#10
    modifiedxTrain10 = poly.fit_transform(xTrain10)
    modifiedxValidation10 = poly.fit_transform(xValidation10)
    
    MSE1 = linearRegression(modifiedxTrain1, modifiedxValidation1, yTrain1, yValidation1)
    MSE2 = linearRegression(modifiedxTrain2, modifiedxValidation2, yTrain2, yValidation2)
    MSE3 = linearRegression(modifiedxTrain3, modifiedxValidation3, yTrain3, yValidation3)
    MSE4 = linearRegression(modifiedxTrain4, modifiedxValidation4, yTrain4, yValidation4)    
    MSE5 = linearRegression(modifiedxTrain5, modifiedxValidation5, yTrain5, yValidation5)
    MSE6 = linearRegression(modifiedxTrain6, modifiedxValidation6, yTrain6, yValidation6)
    MSE7 = linearRegression(modifiedxTrain7, modifiedxValidation7, yTrain7, yValidation7)
    MSE8 = linearRegression(modifiedxTrain8, modifiedxValidation8, yTrain8, yValidation8)
    MSE9 = linearRegression(modifiedxTrain9, modifiedxValidation9, yTrain9, yValidation9)
    MSE10 = linearRegression(modifiedxTrain10, modifiedxValidation10, yTrain10, yValidation10)
    
    avgMSE = (MSE1 + MSE2 + MSE3 + MSE4 + MSE5 + MSE6 + MSE7 + MSE8 + MSE9 + MSE10) / 10
    MSElist.append(avgMSE)
    if avgMSE < minMSE:
        minMSE = avgMSE
        bestDegree = i    
    print("with degree " + str(i) + ", the avgMSE = " + str(avgMSE))


# test
poly = PolynomialFeatures(degree=bestDegree)
modifiedxTrain = poly.fit_transform(trainInput)
modifiedxTest = poly.fit_transform(testInput)
testMSE = linearRegression(modifiedxTrain, modifiedxTest, trainTarget, testTarget)
print("MSE for test set with best degree " + str(bestDegree) + " is " + str(testMSE))
        
plt.plot(degreeRange, MSElist)
plt.xlabel("degree of the monomial basis functions")
plt.ylabel("MSE")
plt.show()




"""

# verify part.......................................................

for i in range(1,5):
    poly = PolynomialFeatures(degree=i)
    
# modify dataset
#1
    modifiedxTrain1 = poly.fit_transform(xTrain1)
    modifiedxValidation1 = poly.fit_transform(xValidation1)
#2
    modifiedxTrain2 = poly.fit_transform(xTrain2)
    modifiedxValidation2 = poly.fit_transform(xValidation2)
#3
    modifiedxTrain3 = poly.fit_transform(xTrain3)
    modifiedxValidation3 = poly.fit_transform(xValidation3)
#4
    modifiedxTrain4 = poly.fit_transform(xTrain4)
    modifiedxValidation4 = poly.fit_transform(xValidation4)
#5
    modifiedxTrain5 = poly.fit_transform(xTrain5)
    modifiedxValidation5 = poly.fit_transform(xValidation5)
#6
    modifiedxTrain6 = poly.fit_transform(xTrain6)
    modifiedxValidation6 = poly.fit_transform(xValidation6)
#7
    modifiedxTrain7 = poly.fit_transform(xTrain7)
    modifiedxValidation7 = poly.fit_transform(xValidation7)
#8
    modifiedxTrain8 = poly.fit_transform(xTrain8)
    modifiedxValidation8 = poly.fit_transform(xValidation8)
#9
    modifiedxTrain9 = poly.fit_transform(xTrain9)
    modifiedxValidation9 = poly.fit_transform(xValidation9)
#10
    modifiedxTrain10 = poly.fit_transform(xTrain10)
    modifiedxValidation10 = poly.fit_transform(xValidation10)

# train with modified dataset
#1
    #reg = LinearRegression().fit(modifiedxTrain1, yTrain1)
    reg = linear_model.BayesianRidge().fit(modifiedxTrain1, yTrain1)
    pred = reg.predict(modifiedxValidation1)
    MSE1 = metrics.mean_squared_error(yValidation1, pred)
#2
    #reg = LinearRegression().fit(modifiedxTrain2, yTrain2)
    reg = linear_model.BayesianRidge().fit(modifiedxTrain2, yTrain2)
    pred = reg.predict(modifiedxValidation2)
    MSE2 = metrics.mean_squared_error(yValidation2, pred)
#3
    #reg = LinearRegression().fit(modifiedxTrain3, yTrain3)
    reg = linear_model.BayesianRidge().fit(modifiedxTrain3, yTrain3)
    pred = reg.predict(modifiedxValidation3)
    MSE3 = metrics.mean_squared_error(yValidation3, pred)
#4
    #reg = LinearRegression().fit(modifiedxTrain4, yTrain4)
    reg = linear_model.BayesianRidge().fit(modifiedxTrain4, yTrain4)
    pred = reg.predict(modifiedxValidation4)
    MSE4 = metrics.mean_squared_error(yValidation4, pred)
#5
    #reg = LinearRegression().fit(modifiedxTrain5, yTrain5)
    reg = linear_model.BayesianRidge().fit(modifiedxTrain5, yTrain5)
    pred = reg.predict(modifiedxValidation5)
    MSE5 = metrics.mean_squared_error(yValidation5, pred)
#6
    #reg = LinearRegression().fit(modifiedxTrain6, yTrain6)
    reg = linear_model.BayesianRidge().fit(modifiedxTrain6, yTrain6)
    pred = reg.predict(modifiedxValidation6)
    MSE6 = metrics.mean_squared_error(yValidation6, pred)
#7
    #reg = LinearRegression().fit(modifiedxTrain7, yTrain7)
    reg = linear_model.BayesianRidge().fit(modifiedxTrain7, yTrain7)
    pred = reg.predict(modifiedxValidation7)
    MSE7 = metrics.mean_squared_error(yValidation7, pred)
#8
    #reg = LinearRegression().fit(modifiedxTrain8, yTrain8)
    reg = linear_model.BayesianRidge().fit(modifiedxTrain8, yTrain8)
    pred = reg.predict(modifiedxValidation8)
    MSE8 = metrics.mean_squared_error(yValidation8, pred)
#9
    #reg = LinearRegression().fit(modifiedxTrain9, yTrain9)
    reg = linear_model.BayesianRidge().fit(modifiedxTrain9, yTrain9)
    pred = reg.predict(modifiedxValidation9)
    MSE9 = metrics.mean_squared_error(yValidation9, pred)
#10
    #reg = LinearRegression().fit(modifiedxTrain10, yTrain10)
    reg = linear_model.BayesianRidge().fit(modifiedxTrain10, yTrain10)
    pred = reg.predict(modifiedxValidation10)
    MSE10 = metrics.mean_squared_error(yValidation10, pred)

    avgMSE = (MSE1 + MSE2 + MSE3 + MSE4 + MSE5 + MSE6 + MSE7 + MSE8 + MSE9 + MSE10) / 10
    print("with degree " + str(i) + ", the avgMSE = " + str(avgMSE))

"""