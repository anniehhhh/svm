import pickle  # For loading the SVM model
from sklearn.feature_extraction.text import TfidfVectorizer  # Example feature extraction (modify as needed)
from sklearn.svm import SVC  # SVM classifier (modify as needed)
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model_path = 'svm_model.pkl' 
with open(model_path, 'rb') as model_file:
    svm_model = pickle.load(model_file)


csv_file_path = 'sampleData.csv' 
input_data = pd.read_csv(csv_file_path)


label_encoder = LabelEncoder()

custom_mapping = {"YES": 1, "NO": 0}

label_encoder.fit(list(custom_mapping.keys()))
input_data['ADHD of blood relative?'] = label_encoder.fit_transform(input_data['ADHD of blood relative?'])

data = input_data[['Age','Attentive Score','Hyperactivity Score','Game1 tries','Game1 time','Game2 tries','Game2 time','ADHD of blood relative?']]

prediction = svm_model.predict(data)

# Interpret the prediction
if prediction == 1:
    print("ADHD ACHE")
else:
    print("ADHD NEI.")

"""
if prediction[0] == 1:
    print("The data is prompted.")
else:
    print("The data is not prompted.")
"""