import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score, precision_score, recall_score,classification_report
data=pd.read_csv('Iris - Iris.csv')
#print(data.head(10))



def calculate_prior(df,Y):
    classes=sorted(list(df[Y].unique()))
    prior=[]
    for i in  classes:
        prior.append(len(df[df[Y]==i])/len(df))
    return prior
def calculate_likelihood(df,feat_name,feat_val,Y,label):
    feat=list(df.columns)
    df=df[df[Y]==label]
    mean,std=df[feat_name].mean(),df[feat_name].std()
    p_x_given_y=(1/(np.sqrt(2*np.pi)*std))* np.exp(-((feat_val-mean)**2/(2*std**2)))
    return p_x_given_y

def naive_bayes(df,X,Y):
    features=list(df.columns)[:-1]

    prior=calculate_prior(df,Y)

    y_pred = []

    for x in X:
        labels=sorted(list(df[Y].unique()))
        likelihood=[1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j]*= calculate_likelihood(df,features[i],x[i],Y,labels[j])

        post_prob=[1]*len(labels)
        for j in range(len(labels)):
            post_prob[j]=likelihood[j]*prior[j]
        

        y_pred.append(np.argmax(post_prob))
    return np.array(y_pred)

from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=.2,random_state=40)


x_test=test.iloc[:,:-1].values
y_test=test.iloc[:,-1].values
y_pred=naive_bayes(train,X=x_test,Y="Species")

print(confusion_matrix(y_test,y_pred))
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy   :", accuracy)
precision = precision_score(y_test, y_pred)
print("Precision :", precision)
recall = recall_score(y_test, y_pred)
print("Recall    :", recall)
F1_score = f1_score(y_test, y_pred)
print("F1-score  :", F1_score)
#print(f1_score(y_test,y_pred))



