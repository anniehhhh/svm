import numpy as np
from collections import Counter
import pandas as pd

#calculates euclidean distance
def distance(p1,p2):
    d=np.sqrt(np.sum((p1-p2)**2))
    return d
#k-nearest neighbors
def knn(training_data,new_point,k):
    distance_set=[]              #keeps the distances of new_point from each point in training_dataset
    for p in training_data:      #for each row p in training_dataset
        d=distance(p[:-1],new_point[:-1])   #taking 3 points x,y,z columns excluding the last col which is classname
        distance_set.append((p,d))  #adding to the distance_set list , the sublist containing list of x,y,z,class and the distance
    distance_set.sort(key=lambda x:x[1])  #sort the list by the key where lamda is an anonymous function 
                                          #which takes x as input and returns x[1] means the distance part of the distance list
    neighbors=[item[0] for item in distance_set[:k]]    #stores first 3,5,7 points accordingly
    return neighbors

#classifies a new point based o majority class among k neighbors
def classify(neighbors):
    class_counter= Counter(neighbor[-1] for neighbor in neighbors) #list contains all the class of neighbors
    most_common_class=class_counter.most_common(1)[0][0]  #stores the most frequent class(0) among all,
                                                          #passing arguements to most_common(1) returns 1 value at index(0)
    return most_common_class

data=pd.read_csv('knn_Data_1.csv',header=None)

training_data=data.iloc[:50].values #first 50 data points
testing_data=data.iloc[50:].values  #last 10 data points

k_set=[3,5,7]  #values of k

for k in k_set:  #for each value of k
    correct_predictions=0
    for test_p in testing_data:  #each row in testing_dataset
        neighbors=knn(training_data,test_p,k)   #finding knn
        predicted_class=classify(neighbors)    #predict most frequent class
        print(f'predictions for k ={k} for {test_p}: {predicted_class}')  #prints predicted classes for each testing set
        if predicted_class==test_p[-1]:    #check if given class equals predicted class
            correct_predictions +=1        #no of correct predictions
    accuracy=correct_predictions/len(testing_data)*100  #no of correct predictions/total testing data * 100
    print(f'Accuracy for k = {k}:{accuracy:.2f}%')      #prints accuracy


