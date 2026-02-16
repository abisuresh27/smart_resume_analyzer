import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load Datasets
# -----------------------------
# Resume dataset: must have columns -> Resume, Role
resume_data = pd.read_csv("resume_dataset.csv")

# JD dataset: must have columns -> JobDescription, Role
jd_data = pd.read_csv("job_descriptions.csv")

# -----------------------------
# 2. Preprocess Text
# -----------------------------
def preprocess(text):
    text = re.sub(r'[^a-zA-Z ]', '', str(text))  # remove special chars
    return text.lower()

resume_data["Cleaned_Text"] = resume_data["Resume"].apply(preprocess)
jd_data["Cleaned_Text"] = jd_data["JobDescription"].apply(preprocess)

# -----------------------------
# 3. Combine Both Datasets
# -----------------------------
combined = pd.concat([
    resume_data[["Role", "Cleaned_Text"]],
    jd_data[["Role", "Cleaned_Text"]]
])

# -----------------------------
# 4. Vectorize Text
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(combined["Cleaned_Text"])
y = combined["Role"]

# -----------------------------
# 5. Train Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("✅ Training complete")
print("📊 Accuracy on test set:", model.score(X_test, y_test))

# -----------------------------
# 6. Save Model + Vectorizer
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("💾 Model and vectorizer saved as model.pkl and vectorizer.pkl")
