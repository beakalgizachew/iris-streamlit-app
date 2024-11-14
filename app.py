# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
st.title("Iris Dataset Classifier")
st.write("This is a simple Streamlit app that classifies Iris flower species.")

@st.cache_data
def load_data():
    return sns.load_dataset("iris")

df = load_data()

# Display the dataset
if st.checkbox("Show dataset"):
    st.write(df)

# Data visualization
st.subheader("Data Visualization")
feature = st.selectbox("Select feature to visualize", df.columns[:-1])
fig, ax = plt.subplots()
sns.histplot(df[feature], kde=True, ax=ax)
st.pyplot(fig)

# Train a model
st.subheader("Train a Model")
test_size = st.slider("Test set size", 0.1, 0.9, 0.3)

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Model evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model accuracy: {accuracy * 100:.2f}%")

# Predict species
st.subheader("Make Predictions")
sepal_length = st.slider("Sepal length", float(df["sepal_length"].min()), float(df["sepal_length"].max()))
sepal_width = st.slider("Sepal width", float(df["sepal_width"].min()), float(df["sepal_width"].max()))
petal_length = st.slider("Petal length", float(df["petal_length"].min()), float(df["petal_length"].max()))
petal_width = st.slider("Petal width", float(df["petal_width"].min()), float(df["petal_width"].max()))

if st.button("Predict"):
    prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.write(f"The predicted species is: {prediction[0]
