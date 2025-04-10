import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics

# --------------------------
# Session state setup
# --------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "data" not in st.session_state:
    st.session_state.data = None
if "model" not in st.session_state:
    st.session_state.model = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Linear Regression"

def go_to(page):
    st.session_state.page = page

# ---------------------------
# 🏠 HOME PAGE
# ---------------------------
if st.session_state.page == "home":
    st.title("Stock Price Prediction App")

    # Dropdown to select dataset
    dataset_option = st.selectbox("📊 Select Dataset", ["None", "Tesla", "Reliance", "Apple"])
    dataset_files = {
        "Tesla": "C:\Streamlit\TSLA.csv",
        "Reliance": "C:\Streamlit\RELIANCE.NS.csv",
        "APPL": "C:\Streamlit\AAPL.csv"
    }

    if dataset_option != "None":
        try:
            df = pd.read_csv(dataset_files[dataset_option])
            st.session_state.data = df
            st.success(f"✅ Loaded {dataset_option} dataset successfully!")
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
    else:
        uploaded_file = st.file_uploader("📁 Or Upload Dataset (CSV)", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.success("✅ Dataset uploaded successfully!")

    if st.button("🧹 Data Preprocessing"):
        if st.session_state.data is not None:
            go_to("preprocessing")
        else:
            st.warning("⚠️ Please upload or select a dataset first!")

    if st.button("📊 Select Model"):
        go_to("model_selection")

    if st.button("📈 Model Evaluation"):
        if st.session_state.data is not None:
            go_to("model_evaluation")
        else:
            st.warning("⚠️ Please upload or select a dataset first!")

# ---------------------------
# 🧹 DATA PREPROCESSING PAGE
# ---------------------------
elif st.session_state.page == "preprocessing":
    st.title("Data Preprocessing & EDA")

    data = st.session_state.data

    if data is not None:
        st.subheader("Dataset Shape")
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

        st.subheader("🔍 Dataset Preview")
        st.dataframe(data.head())

        st.subheader("📊 Summary Statistics")
        st.write(data.describe())

        st.subheader("❓ Missing Values")
        missing = data.isnull().sum()
        st.write(missing)

        if missing.any():
            st.markdown("Handle Missing Values")
            strategy = st.selectbox("Select missing value strategy", ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"])
            if st.button("Apply Strategy"):
                if strategy == "Drop rows":
                    data = data.dropna()
                elif strategy == "Fill with mean":
                    data = data.fillna(data.mean(numeric_only=True))
                elif strategy == "Fill with median":
                    data = data.fillna(data.median(numeric_only=True))
                elif strategy == "Fill with mode":
                    for col in data.columns:
                        data[col] = data[col].fillna(data[col].mode()[0])
                st.success("✅ Missing values handled!")
                st.session_state.data = data  # Save updated data

        st.subheader("Data Types")
        st.write(data.dtypes)

    else:
        st.warning("⚠️ No dataset uploaded.")

    if st.button("⬅️ Back to Home"):
        go_to("home")


# ---------------------------
# 🤖 MODEL SELECTION PAGE
# ---------------------------
elif st.session_state.page == "model_selection":
    st.title("Model Selection")
    st.write("Choose your prediction model:")

    model_choice = st.radio("Available Models", ["Linear Regression", "One Hot Encoding + Linear Regression"])
    st.session_state.selected_model = model_choice
    st.success(f"✅ You selected: {model_choice}")

    if st.button("⬅️ Back to Home"):
        go_to("home")

    if st.button("➡️ Go to Model Evaluation"):
        if st.session_state.data is not None:
            go_to("model_evaluation")
        else:
            st.warning("⚠️ Please upload a dataset first!")

# ----------------------
# 📈 MODEL EVALUATION PAGE
# ----------------------
elif st.session_state.page == "model_evaluation":
    st.title("📈 Model Evaluation")
    data = st.session_state.data

    if data is not None:
        st.subheader("📊 Uploaded Dataset (Preview):")
        st.dataframe(data.head())

        try:
            features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            X = data[features]
            y = data['Close']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            if st.session_state.selected_model == "Linear Regression":
                model = LinearRegression()
            else:
                categorical_features = []
                numerical_features = features
                preprocessor = ColumnTransformer([
                    ("num", "passthrough", numerical_features),
                    ("cat", OneHotEncoder(), categorical_features)
                ])
                model = Pipeline([
                    ("preprocessor", preprocessor),
                    ("regressor", LinearRegression())
                ])

            model.fit(X_train, y_train)
            st.session_state.model = model

            st.subheader("🎛 Enter Input Values")

            def fancy_input(label, key, default):
                col1, col2, col3 = st.columns([1, 6, 1])
                with col1:
                    st.markdown("### -")
                with col2:
                    return st.number_input(label, value=float(default), key=key)
                with col3:
                    st.markdown("### +")

            open_val = fancy_input("Open", "open", data["Open"].mean())
            high_val = fancy_input("High", "high", data["High"].mean())
            low_val = fancy_input("Low", "low", data["Low"].mean())
            close_val = fancy_input("Close", "close", data["Close"].mean())
            adj_close_val = fancy_input("Adj Close", "adj_close", data["Adj Close"].mean())
            volume_val = fancy_input("Volume", "volume", data["Volume"].mean())

            if st.button("Predict"):
                input_data = np.array([[open_val, high_val, low_val, close_val, adj_close_val, volume_val]])
                prediction = model.predict(input_data)[0]
                st.success(f"✅ Predicted Close Price: {prediction:.2f}")

                y_pred = model.predict(X_test)
                r2 = metrics.r2_score(y_test, y_pred)
                st.info(f"📈 Model R² Accuracy on Test Data: {r2:.4f}")

        except Exception as e:
            st.error(f"❌ Error during processing: {e}")
    else:
        st.warning("No dataset found. Please upload one from the Home page.")

    if st.button("⬅️ Back to Home"):
        go_to("home")
