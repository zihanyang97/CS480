import csv
import numpy as np
from numpy import *
import math
import random
import operator

# each train file has 20 instances
# with 2 attributes

# function to import train data
def importData(file):
    data = []
    csvFile = open(file, "rd")
    csv_reader = csv.reader(csvFile, delimiter=',')
    
    for row in csv_reader:
        tmp = []
        for col in row:
            tmp.append(col)
        data.append(tmp)
        
    return data

# function for linear regression
def linearRegression(k, xTrain, xValidation, yTrain, yValidation):
    # set mse to infinite large value
    MSE = 0
    # get lambdaI
    I = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    lambdaI = k * I
    # get A inverse
    M = np.matrix(xTrain)
    M = M.getT()
    A = dot(M,M.getT()) + lambdaI
    AI = A.getI()
    # get b
    b = zeros((3,1))
    for i in range(0,len(xTrain1)):
        N = np.matrix(xTrain1[i])
        N = N.getT()
        b += float(yTrain1[i][0]) * N
    W = dot(AI,b)
    #print W
    W0 = float(W[0])
    W1 = float(W[1])
    W2 = float(W[2])
    #print W # check W correctness
    for i in range(0, len(xValidation)):
        #print xValidation[i]
        predict = W0 + W1*xValidation[i][1] + W2*xValidation[i][2]
        MSE += (predict - float(yValidation[i][0]))**2
    return 0.5*MSE

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

# add 1 into every X
for i in trainInput:
    i.insert(0, 1)
    i[1] = float(i[1])
    i[2] = float(i[2])
#print trainInput

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

#print len(trainInput)
#print len(trainTarget)


# import test data
testInput = importData("testInput.csv")

for i in testInput:
    i.insert(0, 1)
    i[1] = float(i[1])
    i[2] = float(i[2])

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

# set lambda
lambda_range = np.arange(0.1, 4.1, 0.1)

minMSE = float('inf')
score_list = []
K = 0
for k in lambda_range:
    MSE1 = linearRegression(k, xTrain1, xValidation1, yTrain1, yValidation1)
    MSE2 = linearRegression(k, xTrain2, xValidation2, yTrain2, yValidation2)
    MSE3 = linearRegression(k, xTrain3, xValidation3, yTrain3, yValidation3)
    MSE4 = linearRegression(k, xTrain4, xValidation4, yTrain4, yValidation4)
    MSE5 = linearRegression(k, xTrain5, xValidation5, yTrain5, yValidation5)
    MSE6 = linearRegression(k, xTrain6, xValidation6, yTrain6, yValidation6)
    MSE7 = linearRegression(k, xTrain7, xValidation7, yTrain7, yValidation7)
    MSE8 = linearRegression(k, xTrain8, xValidation8, yTrain8, yValidation8)
    MSE9 = linearRegression(k, xTrain9, xValidation9, yTrain9, yValidation9)
    MSE10 = linearRegression(k, xTrain10, xValidation10, yTrain10, yValidation10)
    avgMSE = (MSE1 + MSE2 + MSE3 + MSE4 + MSE5 + MSE6 + MSE7 + MSE8 + MSE9 + MSE10) / 10
    if avgMSE < minMSE:
        minMSE = avgMSE
        K = k
    print k
    print avgMSE
    #score_list = 

print "best lambda is " + str(K)

result = linearRegression(K, trainInput, testInput, trainTarget, testTarget)

print "MSE using best lambda " + str(K) + " is " + str(result)

"""
A = np.matrix([[1,2], [3,4]])
print(A)
B = tile([5,6], (3,1))
print(A.getT())



A = np.matrix([[1,2],[1,3]])
print A
print A.getT()
print dot(A,A.getT())
print A.getI()


M = np.matrix(xTrain1)
M = M.getT()
print M.getT()
A = dot(M,M.getT())
print A 
print A * 2
print zeros((3,1))

for i in range(0,len(xTrain1)):
    N = np.matrix(xTrain1[i])
    N = N.getT()
    N = float(yTrain1[i][0]) * N
    
print linearRegression(10, xTrain1, xValidation1, yTrain1, yValidation1)
"""