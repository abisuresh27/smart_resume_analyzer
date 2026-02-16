import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text preprocessing
def preprocess(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text

# Skill suggestions
skill_suggestions = {
    "Data Scientist": ["Machine Learning", "Deep Learning", "Statistics", "Python", "SQL"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Node.js"],
    "Android Developer": ["Java", "Kotlin", "Firebase", "Android Studio"],
    "Python Developer": ["Python", "Django", "Flask", "APIs"],
    "Java Developer": ["Core Java", "Spring Boot", "Hibernate"],
}

st.set_page_config(page_title="Smart Resume Skill Gap Analyzer", layout="centered")

st.title("Smart Resume Skill Gap Analyzer")

uploaded_file = st.file_uploader("Upload Resume (TXT file)", type=["txt"])

if uploaded_file is not None:
    resume_text = uploaded_file.read().decode("utf-8")

    st.subheader("Resume Preview")
    st.write(resume_text[:500])

    if st.button("Predict Role"):
        clean_resume = preprocess(resume_text)
        vector = vectorizer.transform([clean_resume])
        prediction = model.predict(vector)[0]

        st.success(f"Predicted Job Role: {prediction}")

        if prediction in skill_suggestions:
            st.subheader("Recommended Skills to Improve")
            for skill in skill_suggestions[prediction]:
                st.write(f"- {skill}")
        else:
            st.write("No skill suggestions available.")
