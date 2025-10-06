import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

# ====================================
# Downloads
# ====================================
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

# ====================================
# Text Preprocessing
# ====================================
lemmatizer = WordNetLemmatizer()
custom_stopwords = set(w for w in stopwords.words("english") if w not in {"no", "not", "nor"})

def lexical_preprocess(text):
    try:
        text = str(text).lower()
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)

        clean_tokens = []
        for word, tag in tagged:
            if word.isalpha() and word not in custom_stopwords:
                if tag.startswith("J"):
                    pos = "a"
                elif tag.startswith("V"):
                    pos = "v"
                elif tag.startswith("N"):
                    pos = "n"
                elif tag.startswith("R"):
                    pos = "r"
                else:
                    pos = "n"
                clean_tokens.append(lemmatizer.lemmatize(word, pos))
        return " ".join(clean_tokens)
    except Exception:
        return ""

# ====================================
# Vectorization
# ====================================
def vectorize_phase(train_texts, test_texts, phase_name):
    try:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            stop_words=None
        )
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        return vectorizer, X_train, X_test
    except Exception as e:
        st.warning(f"Vectorization failed in {phase_name}: {e}")
        return None, None, None

# ====================================
# Model Training and Evaluation
# ====================================
def train_and_eval(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rpt = classification_report(y_test, y_pred, output_dict=True)
    return acc, rpt

# ====================================
# Streamlit App
# ====================================
st.title("🧠 Multi-Phase Text Classification App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of data:", df.head())

    text_col = st.selectbox("Select text column", df.columns)
    label_col = st.selectbox("Select label column", df.columns)

    # Clean text
    st.info("Preprocessing text... please wait ⏳")
    df["clean_text"] = df[text_col].apply(lexical_preprocess)

    X = df["clean_text"]
    y = df[label_col]

    # Train/Test Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    except ValueError as e:
        st.warning(f"⚠️ Using non-stratified split due to: {e}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define Models
    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": SVC(kernel="linear", probability=True),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=20),
    }

    # Simulate “phases”
    train_phases = {"Phase 1": X_train, "Phase 2": X_train.sample(frac=0.7, random_state=42)}
    test_phases = {"Phase 1": X_test, "Phase 2": X_test.sample(frac=0.7, random_state=42)}

    results = []
    reports = {}

    # Training Loop
    for model_name, model in models.items():
        for phase_name in train_phases.keys():
            try:
                _, Xtr, Xte = vectorize_phase(
                    train_phases[phase_name],
                    test_phases[phase_name],
                    phase_name
                )
                if Xtr is None or Xtr.shape[0] == 0:
                    st.warning(f"⚠️ Skipping {phase_name} - no data after preprocessing.")
                    continue

                acc, rpt = train_and_eval(model, Xtr, Xte, y_train, y_test)
                results.append({"Phase": phase_name, "Model": model_name, "Accuracy": acc})
                reports[(model_name, phase_name)] = rpt

            except ValueError as e:
                if "least populated class" in str(e):
                    st.info(f"Skipping {phase_name} for {model_name} due to class imbalance.")
                else:
                    st.warning(f"⚠️ Training failed for {phase_name}/{model_name}: {e}")

    # Results Table
    if results:
        results_df = pd.DataFrame(results)
        pivot_df = results_df.pivot(index="Phase", columns="Model", values="Accuracy")
        st.subheader("📊 Accuracy Comparison Table")
        st.dataframe(pivot_df.style.format("{:.4f}"))

        # Bar Chart
        st.subheader("📈 Phase-wise Accuracy by Model")
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(pivot_df.index))
        width = 0.18
        for i, model in enumerate(pivot_df.columns):
            ax.bar(x + i * width, pivot_df[model], width=width, label=model)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(pivot_df.index, rotation=25)
        ax.legend()
        ax.set_ylim(0, 1)
        st.pyplot(fig)

    # Prediction Box
    st.subheader("🧾 Test a Custom Statement")
    user_text = st.text_area("Enter text to classify:")
    if st.button("Predict"):
        try:
            cleaned = lexical_preprocess(user_text)
            vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
            vec.fit(X_train)
            X_new = vec.transform([cleaned])

            preds = {}
            for model_name, model in models.items():
                try:
                    preds[model_name] = model.predict(X_new)[0]
                except Exception:
                    preds[model_name] = "⚠️ Model not trained properly"

            st.json(preds)
        except Exception as e:
            st.error(f"Prediction error: {e}")
