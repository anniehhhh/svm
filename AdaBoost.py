# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset from a CSV file
iris_data = pd.read_csv('Iris - Iris.csv')

# Assuming that the CSV file has the following columns: 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', and 'species'
X = iris_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_data['Species']

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a base classifier (e.g., Decision Tree)
base_classifier = DecisionTreeClassifier(max_depth=1)  # Weak learner

# Create an AdaBoost classifier with the base classifier
adaboost_classifier = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=50, random_state=42)

# Fit the classifier to the training data
adaboost_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = adaboost_classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display the classification report
class_report = classification_report(y_test, y_pred, target_names=iris_data['Species'].unique())
print("Classification Report:\n", class_report)
