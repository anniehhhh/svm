# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score, precision_score, recall_score,classification_report
# Load the Iris dataset from a CSV file
adhd_data = pd.read_csv('cleaned_data_1.csv')

adhd_data.dropna(inplace=True)

# Assuming that the CSV file has the following columns: 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', and 'species'
X = adhd_data[['Gender','Age','Handedness','ADHD Index','Med Status','Verbal IQ','Inattentive','Hyper/Impulsive','Performance IQ','Full4 IQ']]
y = adhd_data['DX']

print(X.shape)


print(y.shape)

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Create a Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Fit the classifier to the training data
gnb.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = gnb.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display the classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)


