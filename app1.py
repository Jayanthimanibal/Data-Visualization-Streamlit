import streamlit as st #front end framework library in python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import math

# Title for the app
st.title("Tesla Stock Price Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Overview:")
    st.write(data.head())

    # Data information
    st.write("Dataset Information:")
    st.write(data.info())

    # Data description
    st.write("Dataset Description:")
    st.write(data.describe())

    # Features and target variable
    X = data[['High', 'Low', 'Open', 'Volume']]
    y = data['Close']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    # Model initialization and training
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Display coefficients and intercept
    st.write("Model Coefficients:", regressor.coef_)
    st.write("Model Intercept:", regressor.intercept_)

    # Predictions
    predicted = regressor.predict(X_test)

    # Display actual vs predicted values
    data1 = pd.DataFrame({'Actual': y_test.values, 'Predicted': predicted})
    st.write("Actual vs Predicted:")
    st.write(data1.head(20))

    # Error metrics
    mae = metrics.mean_absolute_error(y_test, predicted)
    mse = metrics.mean_squared_error(y_test, predicted)
    rmse = math.sqrt(mse)
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"Root Mean Squared Error: {rmse}")

    # Bar graph
    graph = data1.head(20)
    st.bar_chart(graph)

    # New data prediction
    st.write("Predict New Data:")
    high = st.number_input("High Price:")
    low = st.number_input("Low Price:")
    open_price = st.number_input("Open Price:")
    volume = st.number_input("Volume:")

    if st.button("Predict"):
        new_data = np.array([[high, low, open_price, volume]])
        predicted_price = regressor.predict(new_data)
        st.write(f"Predicted Close Price: {predicted_price[0]}")
