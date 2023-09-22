# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset from a CSV file (assuming 'iris.csv' contains the dataset)
adhd_data = pd.read_csv('cleaned_data_1.csv')

adhd_data.dropna(inplace=True)

# Assuming that the CSV file has the following columns: 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', and 'species'
X = adhd_data[['Gender','Age','Handedness','ADHD Index','Med Status','Verbal IQ','Inattentive','Hyper/Impulsive','Performance IQ','Full4 IQ']]
y = adhd_data['DX']

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Impute missing values using KNN imputer
imputer = KNNImputer(n_neighbors=5)  # You can adjust the number of neighbors as needed
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)

# Create a KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors as needed

# Fit the classifier to the training data with imputed values
knn_classifier.fit(X_train_imputed, y_train)

# Make predictions on the testing data with imputed values
y_pred = knn_classifier.predict(X_test_imputed)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display the classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)
