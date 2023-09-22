import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score, precision_score, recall_score,classification_report

iris_data = pd.read_csv('Iris - Iris.csv')#load iris dataset

X = iris_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_data['Species']

# Split the dataset into a training set(20%) and a testing set(80%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
#classifiication on testing data vs predicted data
#Gaussian Naive Bayes classifier
gnb = GaussianNB()

#Fits the classifier to the training data
gnb.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = gnb.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display the classification report
#class_report = classification_report(y_test, y_pred)
#print("Classification Report:\n", class_report)

print("confusion matrix for testing data\n")
print(confusion_matrix(y_test,y_pred))
print(len(y_test), len(y_pred))
#classifiication on training data vs predicted data
gnb2 = GaussianNB()

# Fit the classifier to the training data
gnb2.fit(X_train, y_train)

#predictions on the training data
y_pred2 = gnb.predict(X_train)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_train, y_pred2)
print(f"Accuracy: {accuracy:.2f}")

# Display the classification report
#class_report = classification_report(y_train, y_pred2)
#print("Classification Report:\n", class_report)

print("confusion matrix for training data\n")
print(confusion_matrix(y_train,y_pred2))
print(len(y_train), len(y_pred2))
