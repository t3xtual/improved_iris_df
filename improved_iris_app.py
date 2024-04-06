import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

iris_df = pd.read_csv("iris-species.csv")

iris_df['Label'] = iris_df['Species'].map({'Iris-setosa' : 0, 'Iris-versicolor' : 2, 'Iris-virginica' : 1})

X = iris_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = iris_df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


svc_model = SVC(kernel = 'linear').fit(X_train,y_train)

rf_clf = RandomForestClassifier(n_jobs = -1, n_estimators = 100).fit(X_train, y_train)

log_reg = LogisticRegression(n_jobs = -1).fit(X_train, y_train)

#@st_cache()

def prediction(Model, SepalLength, SepalWidth, PetalLength, PetalWidth):
	predicted = Model.predict(np.array([SepalLength, SepalWidth, PetalLength, PetalWidth]).reshape((1,-1)))[0]
	if predicted == 0:
		return 'Iris-setosa'
	elif predicted == 1:
		return 'Iris-virginica'
	else:
		return 'Iris-versicolor'


st.sidebar.title("Iris Flower Species Prediction App")
s_length = st.sidebar.slider('SepalLength', float(iris_df['SepalLengthCm'].min()),float(iris_df['SepalLengthCm'].max()))
s_width = st.sidebar.slider('SepalWidth', float(iris_df['SepalWidthCm'].min()),float(iris_df['SepalWidthCm'].max()))
p_length = st.sidebar.slider('PetalLength', float(iris_df['PetalLengthCm'].min()),float(iris_df['PetalLengthCm'].max()))
p_width = st.sidebar.slider('PetalWidth', float(iris_df['PetalWidthCm'].min()),float(iris_df['PetalWidthCm'].max()))
classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))
if st.sidebar.button("Predict"):
	st.cache_data.clear()
	st.write(classifier)
	if classifier == "Support Vector Machine":
		species_type = prediction(svc_model, s_length, s_width, p_length, p_width)
		score = svc_model.score(X_train, y_train)
		st.write("Species predicted:", species_type)
		st.write("Accuracy score of this model is:", score)
	elif classifier == "Logistic Regression":
		species_type = prediction(log_reg, s_length, s_width, p_length, p_width)
		score = log_reg.score(X_train, y_train)
		st.write("Species predicted:", species_type)
		st.write("Accuracy score of this model is:", score)
	elif classifier == 'Random Forest Classifier':
		species_type = prediction(rf_clf, s_length, s_width, p_length, p_width)
		score = rf_clf.score(X_train, y_train)
		st.write("Species predicted:", species_type)
		st.write("Accuracy score of this model is:", score)
	else:
		st.sidebar.write("Select a model type!")