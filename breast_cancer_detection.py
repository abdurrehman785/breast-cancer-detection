#imports 
import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
import shap 

#data 
data = pd.read_csv("E:\\ML datasets\\Breast_cancer_dataset.csv")

print(data.head())

#loading data into x_train , y_train and x_test , t_test 
x = data.iloc[:, :-1] # all columns except the last 
#y = data.iloc[:, -1] # only the last column
y = data["diagnosis"]  # correct label
x = data.drop(columns=["id", "diagnosis", "Unnamed: 32"])  # remove unwanted columns



x = pd.get_dummies(x) # has has strings 
print("Missing values in y", y.isna().sum())
x = x[y.notna()]
y = y[y.notna()]

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

print("Training samples:", x_train.shape)
print("Testing samples:", x_test.shape)

model = LogisticRegression(penalty = 'l2', C = 0.01, solver = 'liblinear') 

#explainer = shap.LinearExplainer(model,x_train)
# help from ai 

model.fit(x_train , y_train)
accuracy = model.score(x_test, y_test) 
print("Test accuracy ", accuracy) 

sample = {
    'radius_mean': 17.99,
    'texture_mean': 10.38,
    'perimeter_mean': 122.8,
    'area_mean': 1001.0,
    'smoothness_mean': 0.1184,
    'compactness_mean': 0.2776,
    'concavity_mean': 0.3001,
    'concave points_mean': 0.1471,
    'symmetry_mean': 0.2419,
    'fractal_dimension_mean': 0.07871,
    # Add all required columns...
}

sample_df = pd.DataFrame([sample])
sample_df = pd.get_dummies(sample_df)
sample_df = sample_df.reindex(columns=x_train.columns, fill_value=0)

prediction = model.predict(sample_df)
print("Prediction:", prediction[0])

#shap_values= explainer(sample)
