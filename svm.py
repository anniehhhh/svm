# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset from a CSV file (assuming 'iris.csv' contains the dataset)
adhd_data = pd.read_csv('ADHD Data - Sheet1.csv')

adhd_data.dropna(inplace=True)

# Assuming that the CSV file has the following columns: 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', and 'species'
X = adhd_data[['Username','Age','Gender','Attentive Score','Hyperactivity Score','Game1 tries','Game1 time','Game2 tries','Game2 time','ADHD of blood relative?']]
y = adhd_data['Class']

print(adhd_data .head())
# Encode the target variable 'species' to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values using SimpleImputer (you can choose a different strategy)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

# Fit the classifier to the training data with imputed values
svm_classifier.fit(X_train_imputed, y_train)


# Make predictions on the testing data with imputed values
y_pred = svm_classifier.predict(X_test_imputed)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Decode the predicted labels back to original class names
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Display the classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)


