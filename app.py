# app.py

import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit application layout
st.title("Iris Dataset Classification App")
st.write("This app uses a Random Forest Classifier to predict the species of iris flowers.")

# User input for prediction
st.sidebar.header("Input Features")
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal length", float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()))
    sepal_width = st.sidebar.slider("Sepal width", float(X["sepal width (cm)"].min()), float(X["sepal width (cm)"].max()))
    petal_length = st.sidebar.slider("Petal length", float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()))
    petal_width = st.sidebar.slider("Petal width", float(X["petal width (cm)"].min()), float(X["petal width (cm)"].max()))
    data = {"sepal length (cm)": sepal_length, "sepal width (cm)": sepal_width, 
            "petal length (cm)": petal_length, "petal width (cm)": petal_width}
    return pd.DataFrame(data, index=[0])

df = user_input_features()

# Display user input
st.subheader("User Input features")
st.write(df)

# Prediction
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Display prediction
st.subheader("Prediction")
st.write(iris.target_names[prediction][0])

st.subheader("Prediction Probability")
st.write(prediction_proba)

st.subheader("Model Accuracy")
st.write(f"{accuracy * 100:.2f}%")

st.subheader("Modified")
st.write(prediction_proba*3)

