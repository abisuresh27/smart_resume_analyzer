import streamlit as st
import pickle
import re
import PyPDF2
import docx
from sklearn.metrics.pairwise import cosine_similarity

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text preprocessing
def preprocess(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text

# Skill suggestions / roadmap
roadmap = {
    "python": "Practice coding challenges and build small projects.",
    "sql": "Learn database basics and practice queries.",
    "machine learning": "Study ML algorithms and try Kaggle datasets.",
    "django": "Build a simple web app using Django.",
    "flask": "Create APIs with Flask.",
    "react": "Learn React basics and build a frontend project.",
    "java": "Strengthen Core Java and explore Spring Boot.",
}

# Helper function to extract text from uploaded files
def extract_text(uploaded_file):
    text = ""
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    return text

# Streamlit UI
st.set_page_config(page_title="Smart Resume Analyzer", layout="centered")
st.title("Smart Resume Skill Gap Analyzer")

# Upload Resume
resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
resume_text = ""
if resume_file is not None:
    resume_text = extract_text(resume_file)
    st.subheader("Resume Preview")
    st.write(resume_text[:500])

# Upload Job Description
job_file = st.file_uploader("Upload Job Description", type=["pdf", "docx", "txt"])
job_text = ""
if job_file is not None:
    job_text = extract_text(job_file)
    st.subheader("Job Description Preview")
    st.write(job_text[:500])

# Analysis Button
if resume_text and job_text and st.button("Analyze Resume vs Job Description"):
    clean_resume = preprocess(resume_text)
    clean_job = preprocess(job_text)

    resume_vector = vectorizer.transform([clean_resume])
    job_vector = vectorizer.transform([clean_job])

    # Matching percentage
    similarity = cosine_similarity(resume_vector, job_vector)[0][0]
    match_percentage = round(similarity * 100, 2)
    st.info(f"Resume matches {match_percentage}% with the Job Description")

    # Missing skills
    job_skills = [skill.lower() for skill in roadmap.keys()]
    missing = [skill for skill in job_skills if skill not in clean_resume]

    if missing:
        st.warning("Missing Skills:")
        for skill in missing:
            st.write(f"- {skill}")

        # Roadmap
        st.subheader("Roadmap to Improve")
        for skill in missing:
            if skill in roadmap:
                st.write(f"- {skill.capitalize()}: {roadmap[skill]}")
    else:
        st.success("No major skill gaps detected! 🚀")
