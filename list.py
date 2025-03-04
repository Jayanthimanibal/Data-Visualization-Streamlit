import streamlit as st

# Sample list
my_list = ["Apple", "Banana", "Cherry", "Date"]

st.title("List Example in Streamlit")

# Display the list
st.write("Original List:", my_list)

# Adding an item to the list
new_item = st.text_input("Enter a new fruit:")
if st.button("Add to List"):
    my_list.append(new_item)
    st.write("Updated List:", my_list)
