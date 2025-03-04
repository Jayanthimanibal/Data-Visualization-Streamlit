import streamlit as st
from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive 😊"
    elif sentiment < 0:
        return "Negative 😞"
    else:
        return "Neutral 😐"

def main():
    st.set_page_config(page_title="AI-Powered Sentiment Analyzer", page_icon="🧠", layout="centered")
    st.title("AI-Powered Sentiment Analyzer")
    st.write("Enter a sentence or paragraph to analyze its sentiment.")
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About", "Contact"])
    
    if page == "Home":
        st.subheader("Analyze Sentiment")
        user_input = st.text_area("Enter text here:")
        if st.button("Analyze"):
            result = analyze_sentiment(user_input)
            st.subheader("Sentiment Result:")
            st.write(result)
    
    elif page == "About":
        st.subheader("About")
        st.write("This AI-powered tool analyzes text sentiment using Natural Language Processing (NLP).")
        st.write("Sentiment analysis helps in understanding the emotional tone behind a body of text. It's commonly used in various applications such as analyzing social media posts, product reviews, and customer feedback.")
        st.write("### Features:")
        st.write("- Easy-to-use interface")
        st.write("- Real-time sentiment analysis")
        st.write("- Supports multiple languages")
        st.write("- Provides sentiment polarity and subjectivity scores")
        st.write("### How It Works:")
        st.write("The sentiment analyzer uses the TextBlob library, which is built on top of NLTK and provides simple APIs for NLP tasks. The tool analyzes the input text and returns a sentiment polarity score (ranging from -1 to 1) and a subjectivity score (ranging from 0 to 1). A positive polarity indicates positive sentiment, a negative polarity indicates negative sentiment, and a polarity of 0 indicates neutral sentiment.")
        st.write("### Benefits:")
        st.write("- Understand the emotional tone of your text")
        st.write("- Gain insights into customer opinions and feedback")
        st.write("- Improve your content by analyzing its sentiment")
    
    elif page == "Contact":
        st.subheader("Contact")
        st.write("For inquiries, email us at: jayanthimanibal@gmail.com")
        st.write("📞 Phone: +123 456 7890")
        st.write("🌐 Website: [Visit Here](https://huggingface.co/blog/sentiment-analysis-python)")
        st.write("📍 Address: 123 Rattha Tek Meadows, Solinganallur, Chennai-600119")
        st.write("💬 Follow us on [Twitter](https://twitter.com/OpenAI) | [LinkedIn](https://linkedin.com/company/openai))")
    
if __name__ == "__main__":
    main()

