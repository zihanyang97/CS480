import csv
import numpy as np
import math
import random
import operator
import matplotlib.pyplot as plt

"""
from sklearn import KNeighborsClassifier
from sklearn import metrics

"""

# each train file has 100 instances
# with 64 attributes

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

# function to calculate Euclidean distance
def euclidean(data1, data2):
    ret = 0
    for i in range(0, 64):
        ret += (float(data1[i]) + float(data2[i]))**2
    return math.sqrt(ret)

# function about k nearest neighbour
def knn(k, xTrain, xValidation, yTrain, yValidation):
    matches = 0
    for i in range(0, len(xValidation)):
        # label[0] represents # of 5, label[1] represents # of 6
        label = [0,0]
        distList = []
        result = 0
        correctLabel = yValidation[i][0]
        for j in range(0, len(xTrain)):
            dist = euclidean(xValidation[i], xTrain[j])
            distList.append([dist, yTrain[j][0]])
            #print(yTrain[j])
        # sort from small to large
        #print "----------"
        #print distList
        #print "before sort"
        random.shuffle(distList)
        distList.sort(key = operator.itemgetter(0))
        #print "after sort"
        #print distList
        #print "---------"
        
        for l in range(0, k):
            #print(l)
            #print(distList[l][1])
            if distList[l][1] == "5":
                label[0] += 1
            elif distList[l][1] == "6":
                label[1] += 1
        
        if label[0] > label[1]:
            result = 5
        elif label[0] < label[1]:
            result = 6
        else:
            result = random.choice([5,6])
        #print(result)
        #print(correctLabel)
        
        if int(result) == int(correctLabel):
            #print("matches")
            matches += 1
    #print(matches)
    return matches
                



# import train data
trainData = []
trainData += importData("trainData1.csv")
trainData += importData("trainData2.csv")
trainData += importData("trainData3.csv")
trainData += importData("trainData4.csv")
trainData += importData("trainData5.csv")
trainData += importData("trainData6.csv")
trainData += importData("trainData7.csv")
trainData += importData("trainData8.csv")
trainData += importData("trainData9.csv")
trainData += importData("trainData10.csv")

# import train labels
trainLabel = []
trainLabel += importData("trainLabels1.csv")
trainLabel += importData("trainLabels2.csv")
trainLabel += importData("trainLabels3.csv")
trainLabel += importData("trainLabels4.csv")
trainLabel += importData("trainLabels5.csv")
trainLabel += importData("trainLabels6.csv")
trainLabel += importData("trainLabels7.csv")
trainLabel += importData("trainLabels8.csv")
trainLabel += importData("trainLabels9.csv")
trainLabel += importData("trainLabels10.csv")


#import test data
testData = importData("testData.csv")

#import test labels
testLabel = importData("testLabels.csv")

"""
score = {}
score_list = []
for k in k_range:
    knn = KNghborsClassifier(n_neighbors = k)
    knn.fit(trainData, trainLabel)
    y_pred = knn.predict(testData)
    score[k] = metrics.accuracy_score(testLabel, y_pred)
    score_list.append(metrics.accuracy_score(testLabel, y_pred))
    
print score

"""
# split dataset for 10-cross validation

# 1
xValidation1 = trainData[0:100]
yValidation1 = trainLabel[0:100]
xTrain1 = trainData[100:1000]
yTrain1 = trainLabel[100:1000]

# 2 
xValidation2 = trainData[100:200]
yValidation2 = trainLabel[100:200]
xTrain2 = trainData[0:100] + trainData[200:1000]
yTrain2 = trainLabel[0:100] + trainLabel[200:1000]

# 3
xValidation3 = trainData[200:300]
yValidation3 = trainLabel[200:300]
xTrain3 = trainData[0:200] + trainData[300:1000]
yTrain3 = trainLabel[0:200] + trainLabel[300:1000]

# 4
xValidation4 = trainData[300:400]
yValidation4 = trainLabel[300:400]
xTrain4 = trainData[0:300] + trainData[400:1000]
yTrain4 = trainLabel[0:300] + trainLabel[400:1000]

# 5
xValidation5 = trainData[400:500]
yValidation5 = trainLabel[400:500]
xTrain5 = trainData[0:400] + trainData[500:1000]
yTrain5 = trainLabel[0:400] + trainLabel[500:1000]

# 6
xValidation6 = trainData[500:600]
yValidation6 = trainLabel[500:600]
xTrain6 = trainData[0:500] + trainData[600:1000]
yTrain6 = trainLabel[0:500] + trainLabel[600:1000]

# 7
xValidation7 = trainData[600:700]
yValidation7 = trainLabel[600:700]
xTrain7 = trainData[0:600] + trainData[700:1000]
yTrain7 = trainLabel[0:600] + trainLabel[700:1000]

# 8
xValidation8 = trainData[700:800]
yValidation8 = trainLabel[700:800]
xTrain8 = trainData[0:700] + trainData[800:1000]
yTrain8 = trainLabel[0:700] + trainLabel[800:1000]

# 9
xValidation9 = trainData[800:900]
yValidation9 = trainLabel[800:900]
xTrain9 = trainData[0:800] + trainData[900:1000]
yTrain9 = trainLabel[0:800] + trainLabel[900:1000]

# 10
xValidation10 = trainData[900:1000]
yValidation10 = trainLabel[900:1000]
xTrain10 = trainData[0:900]
yTrain10 = trainLabel[0:900]

# 10-fold cross validation to find k

K = 0
bestAvg = 0
k_range = range(1,31)
score_list = []

for i in k_range:
    print(i)
    acc1 = knn(i, xTrain1, xValidation1, yTrain1, yValidation1)
    print "acc1 = " + str(acc1)
    acc2 = knn(i, xTrain2, xValidation2, yTrain2, yValidation2)
    print "acc2 = " + str(acc2)  
    acc3 = knn(i, xTrain3, xValidation3, yTrain3, yValidation3)
    print "acc3 = " + str(acc3)   
    acc4 = knn(i, xTrain4, xValidation4, yTrain4, yValidation4)
    print "acc4 = " + str(acc4)  
    acc5 = knn(i, xTrain5, xValidation5, yTrain5, yValidation5)
    print "acc5 = " + str(acc5) 
    acc6 = knn(i, xTrain6, xValidation6, yTrain6, yValidation6)
    print "acc6 = " + str(acc6) 
    acc7 = knn(i, xTrain7, xValidation7, yTrain7, yValidation7)
    print "acc7 = " + str(acc7)
    acc8 = knn(i, xTrain8, xValidation8, yTrain8, yValidation8)
    print "acc8 = " + str(acc8)   
    acc9 = knn(i, xTrain9, xValidation9, yTrain9, yValidation9)
    print "acc9 = " + str(acc9)   
    acc10 = knn(i, xTrain10, xValidation10, yTrain10, yValidation10)
    print "acc10 = " + str(acc10)
    avg = float((acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + acc10) / 10)
    if avg > bestAvg:
        bestAvg = avg
        K = i
    print(avg)
    score_list.append(avg/100)
    
print("---------------result----------------") 
print("best k is " + str(K))

acc = knn(K, trainData, testData, trainLabel, testLabel)

print("accuracy is " + str(float(100*acc/110))+'%')

plt.plot(k_range, score_list)
plt.xlabel("value of K for knn")
plt.ylabel("testing accuracy")
plt.show()

