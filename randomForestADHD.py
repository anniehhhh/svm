'''
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset from a CSV file
adhd_data = pd.read_csv('cleaned_data_1.csv')

# Assuming that the CSV file has the following columns: 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', and 'species'
X = adhd_data.drop('DX',axis=1)
y = adhd_data['DX']
adhd_data.dropna(inplace=True)
# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display the classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)
'''
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset from a CSV file
adhd_data = pd.read_csv('cleaned_data_1.csv')
adhd_data.dropna(axis=0, inplace=True)
# Assuming that the CSV file has the following columns: 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', and 'species'
X = adhd_data.drop('DX',axis=1)
y = adhd_data['DX']


# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display the classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)
