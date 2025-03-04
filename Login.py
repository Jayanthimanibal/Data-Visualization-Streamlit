import streamlit as st
st.title("Login Page")
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Submit"):
    st.success(f"Welcome, {username}!")
