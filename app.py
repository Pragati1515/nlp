# =========================================
# Imports
# =========================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# NLTK setup
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

lemmatizer = WordNetLemmatizer()
custom_stopwords = set(
    w for w in stopwords.words("english")
    if w not in ["not", "no", "nor", "against", "is", "are", "be", "been"]
)
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# =========================================
# Text Processing Functions
# =========================================
def lexical_preprocess(text):
    try:
        tokens = nltk.word_tokenize(str(text).lower())
        cleaned = [
            lemmatizer.lemmatize(w)
            for w in tokens
            if w not in custom_stopwords and w not in string.punctuation and len(w) > 1
        ]
        return " ".join(cleaned) if cleaned else str(text)
    except:
        return str(text)

def syntactic_features(text):
    try:
        tokens = nltk.word_tokenize(str(text))
        pos_tags = [tag for (_, tag) in nltk.pos_tag(tokens)]
        return " ".join(pos_tags)
    except:
        return str(text)

def semantic_features(text):
    try:
        blob = TextBlob(str(text))
        return f"{blob.sentiment.polarity} {blob.sentiment.subjectivity}"
    except:
        return "0 0"

def discourse_features(text):
    try:
        sentences = nltk.sent_tokenize(str(text))
        first_words = [s.split()[0] for s in sentences if len(s.split()) > 0]
        return f"{len(sentences)} {' '.join(first_words)}"
    except:
        return "0"

def pragmatic_features(text):
    try:
        tokens = []
        for w in pragmatic_words:
            count = str(text).lower().count(w)
            tokens.extend([w] * count)
        return " ".join(tokens)
    except:
        return str(text)

# =========================================
# Vectorization (fixed)
# =========================================
def get_vectorizer(phase_name, max_features=3000):
    if phase_name in ["Lexical & Morphological", "Syntactic"]:
        return TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    else:
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=max_features)

# =========================================
# Model Training
# =========================================
def train_and_eval(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    rpt = classification_report(y_test, preds, zero_division=0, output_dict=True)
    return acc, rpt

# =========================================
# Streamlit App
# =========================================
st.title("📰 Fake News Detection: Phase-wise NLP")
st.write("Upload your dataset and compare performance across linguistic phases.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        text_col = st.selectbox("Select TEXT column", df.columns)
        target_col = st.selectbox("Select TARGET column", df.columns)

        data = df[[text_col, target_col]].dropna().copy()
        data.columns = ["text", "target"]

        if data["target"].dtype == object:
            data["target"] = LabelEncoder().fit_transform(data["target"].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            data["text"].astype(str),
            data["target"],
            test_size=0.2,
            random_state=42,
            stratify=data["target"]
        )

        # Preprocess once per phase
        preprocessors = {
            "Lexical & Morphological": lexical_preprocess,
            "Syntactic": syntactic_features,
            "Semantic": semantic_features,
            "Discourse": discourse_features,
            "Pragmatic": pragmatic_features,
        }

        models = {
            "Naive Bayes": MultinomialNB(),
            "SVM": SVC(kernel="linear", probability=True),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(max_depth=20),
        }

        results = []
        trained_models = {}
        vectorizers = {}

        for phase_name, func in preprocessors.items():
            st.write(f"Processing phase: **{phase_name}** ...")
            Xtr_prep = X_train.apply(func)
            Xte_prep = X_test.apply(func)

            vectorizer = get_vectorizer(phase_name)
            Xtr_vec = vectorizer.fit_transform(Xtr_prep)
            Xte_vec = vectorizer.transform(Xte_prep)

            vectorizers[phase_name] = vectorizer

            for model_name, model in models.items():
                acc, rpt = train_and_eval(model, Xtr_vec, Xte_vec, y_train, y_test)
                results.append({"Phase": phase_name, "Model": model_name, "Accuracy": acc})
                trained_models[(model_name, phase_name)] = model

        results_df = pd.DataFrame(results)
        pivot_df = results_df.pivot(index="Phase", columns="Model", values="Accuracy")

        st.subheader("📊 Accuracy Comparison")
        st.dataframe(pivot_df.style.format("{:.4f}"))

        fig, ax = plt.subplots(figsize=(9,5))
        x = np.arange(len(pivot_df.index))
        width = 0.18
        for i, model in enumerate(pivot_df.columns):
            ax.bar(x + i*width, pivot_df[model], width, label=model)
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(pivot_df.index, rotation=20)
        ax.set_ylim(0, 1)
        ax.legend()
        st.pyplot(fig)

        # =====================================
        # New Text Prediction (fixed)
        # =====================================
        st.subheader("📝 Predict New Text")
        new_text = st.text_area("Enter text to analyze:")
        if st.button("Predict"):
            if new_text.strip():
                results_list = []
                for (model_name, phase_name), model in trained_models.items():
                    vectorizer = vectorizers[phase_name]
                    func = preprocessors[phase_name]
                    new_vec = vectorizer.transform([func(new_text)])
                    pred = model.predict(new_vec)_
