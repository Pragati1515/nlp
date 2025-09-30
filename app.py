# app.py
import streamlit as st
import pandas as pd
import numpy as np
import nltk, string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

# ============================
# Preprocessing
# ============================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def lexical_preprocess(text):
    try:
        tokens = nltk.word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w not in string.punctuation]
        return " ".join(tokens) if tokens else "empty"
    except:
        return "empty"

def syntactic_features(text):
    try:
        tokens = nltk.word_tokenize(text)
        pos_tags = [tag for (_, tag) in nltk.pos_tag(tokens)]
        return " ".join(pos_tags) if pos_tags else "empty"
    except:
        return "empty"

def semantic_features(text):
    try:
        blob = TextBlob(text)
        return f"{blob.sentiment.polarity} {blob.sentiment.subjectivity}"
    except:
        return "0 0"

def discourse_features(text):
    try:
        sentences = nltk.sent_tokenize(text)
        first_words = [s.split()[0] for s in sentences if len(s.split()) > 0]
        return f"{len(sentences)} {' '.join(first_words)}" if sentences else "0"
    except:
        return "0"

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

def pragmatic_features(text):
    try:
        tokens = []
        for w in pragmatic_words:
            count = text.lower().count(w)
            tokens.extend([w] * count)
        return " ".join(tokens) if tokens else "none"
    except:
        return "none"

# ============================
# Safe Vectorizer
# ============================
def safe_vectorize(vectorizer, data, phase_name):
    try:
        X_vec = vectorizer.fit_transform(data)
        if X_vec.shape[1] == 0:  # empty vocab fallback
            raise ValueError("Empty vocabulary")
        return X_vec
    except Exception as e:
        st.warning(f"Feature extraction failed in {phase_name}: {e}. Using fallback dummy feature.")
        return CountVectorizer().fit_transform(["empty" for _ in data])

# ============================
# Train & Evaluate
# ============================
def train_and_eval(model, X_features, y, name):
    try:
        stratify_flag = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42, stratify=stratify_flag
        )

        # Some models (DecisionTree) need dense input
        need_dense = isinstance(model, DecisionTreeClassifier)
        X_train_in = X_train.toarray() if need_dense and hasattr(X_train, "toarray") else X_train
        X_test_in = X_test.toarray() if need_dense and hasattr(X_test, "toarray") else X_test

        model.fit(X_train_in, y_train)
        preds = model.predict(X_test_in)

        acc = accuracy_score(y_test, preds)
        return acc, classification_report(y_test, preds, zero_division=0, output_dict=True)
    except Exception as e:
        st.error(f"Training failed for {name}: {e}")
        return 0.0, {}

# ============================
# Streamlit App
# ============================
st.title("📰 Fake News Detection: Phase-wise NLP with Multiple Models")

uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.write("Dataset Preview:", df.head())

        text_col = st.selectbox("Select the TEXT column", df.columns)
        target_col = st.selectbox("Select the TARGET column", df.columns)

        data = df[[text_col, target_col]].dropna().copy()
        data.columns = ["text", "target"]

        # Encode target
        if data["target"].dtype == object or not np.issubdtype(data["target"].dtype, np.number):
            le = LabelEncoder()
            data["target"] = le.fit_transform(data["target"].astype(str))
            st.info(f"Target labels encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        X = data["text"].astype(str)
        y = data["target"]

        # ============================
        # Phases
        # ============================
        X_lexical = X.apply(lexical_preprocess)
        vec_lexical = safe_vectorize(CountVectorizer(), X_lexical, "Lexical")

        X_syntax = X.apply(syntactic_features)
        vec_syntax = safe_vectorize(CountVectorizer(), X_syntax, "Syntactic")

        X_semantic = X.apply(semantic_features)
        vec_semantic = safe_vectorize(TfidfVectorizer(), X_semantic, "Semantic")

        X_discourse = X.apply(discourse_features)
        vec_discourse = safe_vectorize(CountVectorizer(), X_discourse, "Discourse")

        X_pragmatic = X.apply(pragmatic_features)
        vec_pragmatic = safe_vectorize(CountVectorizer(), X_pragmatic, "Pragmatic")

        phases = [
            ("Lexical & Morphological", vec_lexical),
            ("Syntactic", vec_syntax),
            ("Semantic", vec_semantic),
            ("Discourse", vec_discourse),
            ("Pragmatic", vec_pragmatic),
        ]

        # ============================
        # Models
        # ============================
        models = {
            "Naive Bayes": MultinomialNB(),
            "SVM": SVC(kernel="linear", probability=True),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
        }

        # ============================
        # Run Experiments
        # ============================
        results = []
        reports = {}

        for model_name, model in models.items():
            for phase_name, X_vec in phases:
                acc, report = train_and_eval(model, X_vec, y, f"{model_name} - {phase_name}")
                results.append({"Model": model_name, "Phase": phase_name, "Accuracy": acc})
                if report:
                    reports[(model_name, phase_name)] = report

        results_df = pd.DataFrame(results)
        st.subheader("📊 Phase-wise Accuracies")
        st.dataframe(results_df)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_name in results_df["Model"].unique():
            subset = results_df[results_df["Model"] == model_name]
            ax.plot(subset["Phase"], subset["Accuracy"], marker="o", label=model_name)
        ax.set_title("Phase-wise Accuracies by Model")
        ax.set_xlabel("Phase")
        ax.set_ylabel("Accuracy")
        ax.legend()
        st.pyplot(fig)

        # Classification Reports
        with st.expander("Detailed Classification Reports"):
            for (model_name, phase_name), report in reports.items():
                st.write(f"### {model_name} - {phase_name}")
                rpt_df = pd.DataFrame(report).transpose()
                st.dataframe(rpt_df)

    except Exception as e:
        st.error(f"Error processing file: {e}")
