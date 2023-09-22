# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer  # For imputing missing values
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset from a CSV file
adhd_data = pd.read_csv('cleaned_data_1.csv')

# Assuming that the CSV file has the following columns: 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', and 'species'
X = adhd_data.drop('DX',axis=1)
y = adhd_data['DX']

adhd_data.dropna(axis=0, inplace=False)

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Impute missing values (you can choose a different strategy)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Create a HistGradientBoostingClassifier
hist_gb_classifier = HistGradientBoostingClassifier(random_state=42)

# Fit the classifier to the training data
hist_gb_classifier.fit(X_train_imputed, y_train)

# Make predictions on the testing data
y_pred = hist_gb_classifier.predict(X_test_imputed)


"""

#Create a base classifier (e.g., Decision Tree)
base_classifier = DecisionTreeClassifier(max_depth=1)  # Weak learner

# Create an AdaBoost classifier with the base classifier
adaboost_classifier = AdaBoostClassifier(base_estimator=base_classifier, n_estimators=50, random_state=42)

# Fit the classifier to the training data
adaboost_classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = adaboost_classifier.predict(X_test)
"""

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display the classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)


















