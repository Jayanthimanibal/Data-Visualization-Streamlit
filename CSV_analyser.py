import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# App title
st.title("📊 CSV File Analyzer")

# File uploader
uploaded_file = st.file_uploader("C:\Streamlit\iris data set", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv("C:\Streamlit\iris data set")

    # Show DataFrame

    st.write("### Data Preview:")
    st.dataframe(df.head())

    # Show statistics
    st.write("### Basic Statistics:")
    st.write(df.describe())

    # Select column for visualization
    column = st.selectbox("Select a column to plot", df.columns)

    # Plot histogram
    st.write("### Histogram of Selected Column")
    fig, ax = plt.subplots()
    df[column].hist(ax=ax, bins=20, edgecolor='black')
    st.pyplot(fig)

    # Show raw data option
    if st.checkbox("Show Full Dataset"):
        st.write(df)
else:
    st.info("☝️ Please upload a CSV file to get started.")

