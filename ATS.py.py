# ats_resume_app_updated.py

import streamlit as st
import PyPDF2
import spacy
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Preprocessing function
def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# Ranking resumes based on TF-IDF + cosine similarity
def rank_resumes(job_desc, resume_texts):
    documents = [job_desc] + resume_texts
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(documents)
    similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return similarity

# Extract matched skills from job description
def extract_skills(job_desc, resume_text):
    job_tokens = set(job_desc.split())
    resume_tokens = set(resume_text.split())
    matched = job_tokens.intersection(resume_tokens)
    return list(matched)

# Streamlit Web App
st.set_page_config(page_title="ATS Resume Screening", layout="wide")
st.title("📄 AI-based ATS Resume Screening & Ranking System")
st.markdown("Upload resumes and a job description. The app will analyze, rank, and visualize relevance.")

# Inputs
job_desc_input = st.text_area("📝 Enter the Job Description", height=200, placeholder="e.g. Python, SQL, Power BI...")
uploaded_files = st.file_uploader("📤 Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

# Analyze button
if st.button("🚀 Analyze & Rank Resumes"):
    if not job_desc_input or not uploaded_files:
        st.warning("Please provide both job description and resumes.")
    else:
        # Process job description
        job_desc_clean = preprocess(job_desc_input)
        resume_texts, resume_names, raw_texts = [], [], []

        os.makedirs("uploaded_resumes", exist_ok=True)

        with st.spinner("Processing..."):
            for uploaded_file in uploaded_files:
                try:
                    file_path = os.path.join("uploaded_resumes", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    text = extract_text_from_pdf(uploaded_file)
                    clean_text = preprocess(text)
                    resume_texts.append(clean_text)
                    raw_texts.append(text)
                    resume_names.append(uploaded_file.name)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

        # Ranking logic
        similarities = rank_resumes(job_desc_clean, resume_texts)
        ranked_results = sorted(zip(resume_names, similarities, resume_texts, raw_texts), key=lambda x: x[1], reverse=True)

        # Success message
        st.success("✅ Resume ranking completed!")

        # Prepare for visualization
        filtered_results = [(name, score) for name, score, _, _ in ranked_results]

        # Visualize rankings (compact version)
        st.subheader("📊 Match Score Chart")
        short_names = [name[:15] + "..." if len(name) > 18 else name for name, _ in filtered_results]
        scores = [score for _, score in filtered_results]

        fig, ax = plt.subplots(figsize=(6, 4))  # Smaller figure size
        ax.barh(short_names, scores, color='skyblue')
        ax.set_xlabel("Match Score")
        ax.set_ylabel("Resume")
        ax.set_title("Top Resume Matches", fontsize=12)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

        # Display ranked resumes
        st.subheader("📂 Ranked Resumes and Matched Skills")
        for i, (name, score, clean_text, raw_text) in enumerate(ranked_results, 1):
            st.markdown(f"### {i}. **{name}** — Match Score: `{score:.2f}`")
            skills_matched = extract_skills(job_desc_clean, clean_text)
            st.markdown(f"✅ **Matched Skills**: {', '.join(skills_matched[:10]) if skills_matched else 'None'}")

            with st.expander("📖 View Extracted Resume Text"):
                st.write(raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text)

            with st.expander("🧠 View Named Entities (NER)"):
                doc = nlp(raw_text)
                for ent in doc.ents:
                    st.markdown(f"- `{ent.text}` → *{ent.label_}*")

        # Metadata Table
        st.subheader("📋 Resume Metadata")
        file_info = [{"File Name": name, "Match Score": f"{score:.2f}"} for name, score, _, _ in ranked_results]
        st.dataframe(pd.DataFrame(file_info))

        # Download CSV
        df = pd.DataFrame(file_info)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results as CSV", data=csv, file_name="resume_rankings.csv", mime="text/csv")
