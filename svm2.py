#from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle  # For loading the SVM model
from sklearn.feature_extraction.text import TfidfVectorizer 

data = pd.read_csv('ADHD Data - Sheet1.csv')
# drop rows with missing values
data.dropna(inplace=True)

# Drop other columns you don't need
columns_to_drop = ['Username', 'Gender']
data = data.drop(columns=columns_to_drop)

# Convert 'ADHD of blood relative?' to numeric using label encoding
label_encoder = LabelEncoder()
data['ADHD of blood relative?'] = label_encoder.fit_transform(data['ADHD of blood relative?'])

X = data.drop('Class', axis=1)  # Features (all columns except 'DX')
y = data['Class']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



svm_classifier = SVC(kernel='linear', C=1.0)  # Linear kernel
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred) 

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)

with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm_classifier, model_file)