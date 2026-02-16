import streamlit as st
import pickle
import re
import PyPDF2
import docx
import random
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load trained model + vectorizer
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -----------------------------
# Preprocess function
# -----------------------------
def preprocess(text):
    text = re.sub(r'[^a-zA-Z ]', '', str(text))
    return text.lower()

# -----------------------------
# Roadmap suggestions
# -----------------------------
roadmap = {
    "python": "Practice coding challenges and build ML projects.",
    "sql": "Learn database basics and practice queries.",
    "machine learning": "Study ML algorithms and try Kaggle datasets.",
    "deep learning": "Build neural networks using TensorFlow or PyTorch.",
    "tensorflow": "Create deep learning models with TensorFlow.",
    "pytorch": "Experiment with PyTorch for advanced ML tasks.",
    "aws": "Deploy ML models on AWS SageMaker.",
    "azure": "Explore Azure ML Studio for deployment.",
    "gcp": "Learn Google Cloud AI tools.",
    "statistics": "Revise probability and statistical inference.",
    "django": "Build a simple web app using Django.",
    "flask": "Create APIs with Flask.",
    "react": "Learn React basics and build a frontend project.",
    "java": "Strengthen Core Java and explore Spring Boot.",
}

# -----------------------------
# Extract text from uploaded files
# -----------------------------
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

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Smart Resume Analyzer", layout="centered")
st.title("📊 Smart Resume Skill Gap Analyzer")

# Upload Resume
resume_file = st.file_uploader("📄 Upload Resume", type=["pdf", "docx", "txt"])
resume_text = ""
if resume_file is not None:
    resume_text = extract_text(resume_file)
    st.subheader("📝 Resume Preview")
    st.write(resume_text[:500])

# Enter Target Job Role
job_role = st.text_input("💼 Enter Target Job Role (e.g., Machine Learning Engineer)")

# -----------------------------
# Analysis
# -----------------------------
if resume_text and job_role and st.button("🔍 Analyze Resume"):
    clean_resume = preprocess(resume_text)

    # Predict role from resume
    resume_vector = vectorizer.transform([clean_resume])
    predicted_role = model.predict(resume_vector)[0]

    st.info(f"📌 Predicted Role from Resume: **{predicted_role}**")
    st.info(f"🎯 Target Role: **{job_role}**")

    # Compare similarity between resume and target role keywords
    job_keywords = job_role.lower()
    job_vector = vectorizer.transform([job_keywords])
    similarity = cosine_similarity(resume_vector, job_vector)[0][0]
    match_percentage = round(similarity * 100, 2)

    st.success(f"✅ Resume matches **{match_percentage}%** with the target role")

    # Missing skills
    job_skills = roadmap.keys()
    missing = [skill for skill in job_skills if skill not in clean_resume]

    if missing:
        st.warning("⚠️ Missing Skills Detected:")
        for skill in missing:
            st.write(f"- {skill.capitalize()}")

        # Attractive Roadmap
        st.subheader("🚀 Roadmap to Improve")
        for skill in missing:
            if skill in roadmap:
                progress = random.randint(20, 60)  # simulate current skill level
                with st.expander(f"📘 Learn {skill.capitalize()}"):
                    st.markdown(f"**{skill.capitalize()}** → {roadmap[skill]}")
                    st.progress(progress / 100)
    else:
        st.success("🎉 No major skill gaps detected! You're ready to apply 🚀")
