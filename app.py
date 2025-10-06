# app.py (Improved Version)
import streamlit as st
import pandas as pd
import numpy as np
import nltk, string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

# ============================
# Download NLTK Data
# ============================
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ============================
# Text Preprocessing Functions
# ============================
def lexical_preprocess(text):
    """Tokenize, clean and lemmatize text"""
    try:
        tokens = nltk.word_tokenize(str(text).lower())
        clean_tokens = [
            lemmatizer.lemmatize(w)
            for w in tokens
            if w not in stop_words and w not in string.punctuation and w.isalpha()
        ]
        return " ".join(clean_tokens)
    except Exception:
        return str(text).lower()

def syntactic_features(text):
    try:
        tokens = nltk.word_tokenize(str(text))
        pos_tags = nltk.pos_tag(tokens)
        tags = [tag for _, tag in pos_tags]
        return " ".join(tags)
    except Exception:
        return "NN"

def semantic_features(text):
    try:
        blob = TextBlob(str(text))
        return f"{blob.sentiment.polarity} {blob.sentiment.subjectivity}"
    except Exception:
        return "0 0"

def discourse_features(text):
    try:
        sentences = nltk.sent_tokenize(str(text))
        return f"{len(sentences)}"
    except Exception:
        return "0"

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

def pragmatic_features(text):
    try:
        t = str(text).lower()
        counts = [t.count(w) for w in pragmatic_words]
        return " ".join([f"{w}:{c}" for w, c in zip(pragmatic_words, counts)])
    except Exception:
        return "none"

# ============================
# Vectorization Helper
# ============================
def fit_vectorizer(train_data, test_data, vectorizer):
    """Fit TF-IDF on train and transform both train/test"""
    try:
        vec = vectorizer.fit(train_data)
        X_train = vec.transform(train_data)
        X_test = vec.transform(test_data)
        return X_train, X_test
    except Exception:
        st.warning("Vectorization failed — check data cleanliness.")
        dummy_vec = TfidfVectorizer().fit(["empty"])
        return dummy_vec.transform(train_data), dummy_vec.transform(test_data)

# ============================
# Model Training Function
# ============================
def train_and_eval(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0, output_dict=True)
    return acc, report

# ============================
# Streamlit UI
# ============================
st.title("📰 Enhanced Fake News Detection - Phase-wise NLP")

uploaded = st.file_uploader("Upload CSV file (with text + target)", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        text_col = st.selectbox("Select text column", df.columns)
        target_col = st.selectbox("Select target column", df.columns)

        df = df[[text_col, target_col]].dropna().copy()
        df.columns = ["text", "target"]

        # Encode labels
        le = LabelEncoder()
        df["target"] = le.fit_transform(df["target"].astype(str))

        X = df["text"].astype(str)
        y = df["target"]

        # Split before vectorizing (important!)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ============================
        # Create Feature Phases
        # ============================
        train_phases = {
            "Lexical & Morphological": X_train_raw.apply(lexical_preprocess),
            "Syntactic": X_train_raw.apply(syntactic_features),
            "Semantic": X_train_raw.apply(semantic_features),
            "Discourse": X_train_raw.apply(discourse_features),
            "Pragmatic": X_train_raw.apply(pragmatic_features),
        }

        test_phases = {
            "Lexical & Morphological": X_test_raw.apply(lexical_preprocess),
            "Syntactic": X_test_raw.apply(syntactic_features),
            "Semantic": X_test_raw.apply(semantic_features),
            "Discourse": X_test_raw.apply(discourse_features),
            "Pragmatic": X_test_raw.apply(pragmatic_features),
        }

        # ============================
        # Define Models
        # ============================
        models = {
            "Naive Bayes": MultinomialNB(),
            "SVM": SVC(kernel="linear", probability=True),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
        }

        # ============================
        # Train + Evaluate
        # ============================
        results, reports = [], {}
        vectorizer = TfidfVectorizer(max_features=5000)

        for phase_name in train_phases.keys():
            X_train_vec, X_test_vec = fit_vectorizer(
                train_phases[phase_name], test_phases[phase_name], vectorizer
            )
            for model_name, model in models.items():
                acc, report = train_and_eval(model, X_train_vec, X_test_vec, y_train, y_test)
                results.append({"Phase": phase_name, "Model": model_name, "Accuracy": acc})
                reports[(model_name, phase_name)] = report

        results_df = pd.DataFrame(results)
        pivot_df = results_df.pivot(index="Phase", columns="Model", values="Accuracy")

        st.subheader("📊 Accuracy Comparison (Phases × Models)")
        st.dataframe(pivot_df.style.format("{:.4f}"))

        # ============================
        # Visualization
        # ============================
        st.subheader("📈 Grouped Bar Chart: Model Accuracy by Phase")
        fig, ax = plt.subplots(figsize=(10, 6))
        phases_list = pivot_df.index.tolist()
        models_list = pivot_df.columns.tolist()
        x = np.arange(len(phases_list))
        width = 0.18

        for i, model_name in enumerate(models_list):
            ax.bar(x + i * width, pivot_df[model_name], width=width, label=model_name)

        ax.set_xticks(x + width * (len(models_list) - 1) / 2)
        ax.set_xticklabels(phases_list, rotation=30)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.legend()
        st.pyplot(fig)

        with st.expander("Detailed Classification Reports"):
            for (model_name, phase_name), report in reports.items():
                st.markdown(f"#### {model_name} - {phase_name}")
                rpt_df = pd.DataFrame(report).transpose()
                st.dataframe(rpt_df)

    except Exception as e:
        st.error(f"Error: {e}")
