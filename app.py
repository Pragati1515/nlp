"""
Streamlit app: Phase-wise NLP pipeline (Fake vs Real detection style)
- No spaCy (uses NLTK + TextBlob)
- Handles string labels (true, false, semi-true...) via LabelEncoder
- Compares 4 classifiers across 5 NLP phases
- Visualizes results and allows CSV download
"""

import streamlit as st
import pandas as pd
import numpy as np
import nltk, re, string
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Ensure necessary NLTK downloads
nltk_downloads = ["punkt", "wordnet", "stopwords", "averaged_perceptron_tagger"]
for pkg in nltk_downloads:
    try:
        nltk.data.find(pkg)
    except Exception:
        try:
            nltk.download(pkg)
        except Exception:
            pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize, sent_tokenize

# App UI
st.set_page_config(page_title="Phase-wise NLP Model Comparison", layout="wide")
st.title("Phase-wise NLP Model Comparison — Naive Bayes, SVM, LogisticRegression, Decision Tree")
st.markdown(
    "Upload a CSV, select the text and target columns, then run. "
    "This compares 5 NLP phases across 4 classifiers and visualizes phase-wise accuracy."
)

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Could not read the CSV. Try a different delimiter/encoding.")
        st.stop()

    st.write("Preview of uploaded data (first 5 rows):")
    st.dataframe(df.head())

    # Column selectors
    text_col = st.selectbox("Select the TEXT column", options=df.columns, index=0)
    target_col = st.selectbox("Select the TARGET column", options=df.columns, index=1)

    if st.button("Run phase-wise training and compare models"):
        # Preprocess dataframe
        data = df[[text_col, target_col]].dropna().copy()
        data.columns = ["text", "target"]

        # Encode string targets
        if data["target"].dtype == object or not np.issubdtype(data["target"].dtype, np.number):
            try:
                le = LabelEncoder()
                data["target"] = le.fit_transform(data["target"])
                st.info(f"Target labels encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
            except Exception as e:
                st.error(f"Could not encode target labels: {e}")
                st.stop()

        X = data["text"].astype(str)
        y = data["target"]

        # Preprocessing utilities
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        def lexical_preprocess(text):
            try:
                tokens = word_tokenize(text.lower())
            except Exception:
                tokens = re.findall(r"\b\w+\b", text.lower())
            tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and t not in string.punctuation]
            return " ".join(tokens)

        def syntactic_features(text):
            try:
                tokens = word_tokenize(text)
                tags = pos_tag(tokens)
                return " ".join([tag for (_, tag) in tags])
            except Exception:
                return ""

        def semantic_features(text):
            try:
                blob = TextBlob(text)
                return f"{blob.sentiment.polarity:.4f} {blob.sentiment.subjectivity:.4f}"
            except Exception:
                return "0.0 0.0"

        discourse_connectives = set(["however", "therefore", "moreover", "furthermore", "but", "and", "because", "so", "although", "meanwhile"])
        def discourse_features(text):
            try:
                sents = sent_tokenize(text)
            except Exception:
                sents = re.split(r'[.!?]+', text)
                sents = [s.strip() for s in sents if s.strip()]
            n_sents = len(sents)
            first_words = " ".join([s.split()[0] for s in sents if len(s.split()) > 0])
            conn_counts = " ".join([w for w in discourse_connectives for _ in range(text.lower().count(w))])
            return f"{n_sents} {first_words} {conn_counts}"

        pragmatic_words = ["must", "should", "might", "could", "will"]
        def pragmatic_features(text):
            tokens = []
            for w in pragmatic_words:
                tokens.extend([w] * text.lower().count(w))
            if "?" in text: tokens.append("?")
            if "!" in text: tokens.append("!")
            return " ".join(tokens) if tokens else ""

        # Compute features
        with st.spinner("Preprocessing text..."):
            try:
                X_lexical = X.apply(lexical_preprocess)
                vec_lexical = CountVectorizer().fit_transform(X_lexical)
                X_syntax = X.apply(syntactic_features)
                vec_syntax = CountVectorizer().fit_transform(X_syntax)
                X_semantic = X.apply(semantic_features)
                vec_semantic = TfidfVectorizer().fit_transform(X_semantic)
                X_discourse = X.apply(discourse_features)
                vec_discourse = CountVectorizer().fit_transform(X_discourse)
                X_pragmatic = X.apply(pragmatic_features)
                vec_pragmatic = CountVectorizer().fit_transform(X_pragmatic)
            except Exception as e:
                st.error(f"Feature extraction failed: {e}")
                st.stop()

        # Phase list
        phases = [
            ("Lexical & Morphological", vec_lexical),
            ("Syntactic", vec_syntax),
            ("Semantic", vec_semantic),
            ("Discourse", vec_discourse),
            ("Pragmatic", vec_pragmatic)
        ]

        # Models
        models = {
            "Naive Bayes": MultinomialNB(),
            "SVM": SVC(probability=True, kernel="linear", random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }

        results = []

        # Stratify check
        stratify_flag = None
        if y.nunique() >= 2 and all(y.value_counts() > 1):
            stratify_flag = y

        # Training loop
        for model_name, model in models.items():
            row = {"Model": model_name}
            for phase_name, X_vec in phases:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_vec, y, test_size=0.2, random_state=42, stratify=stratify_flag
                    )
                except Exception:
                    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

                try:
                    need_dense = isinstance(model, DecisionTreeClassifier)
                    X_train_in = X_train.toarray() if need_dense and hasattr(X_train, "toarray") else X_train
                    X_test_in = X_test.toarray() if need_dense and hasattr(X_test, "toarray") else X_test

                    model.fit(X_train_in, y_train)
                    y_pred = model.predict(X_test_in)
                    acc = accuracy_score(y_test, y_pred)
                except Exception as e:
                    st.warning(f"{model_name} failed on {phase_name}: {e}")
                    acc = np.nan

                row[phase_name] = float(acc)
            results.append(row)

        results_df = pd.DataFrame(results).set_index("Model").T
        st.subheader("Phase-wise Accuracies")
        st.dataframe(results_df.style.format("{:.4f}"))

        # Plot
        import matplotlib.pyplot as plt
        phases_names = results_df.index.tolist()
        models_names = results_df.columns.tolist()
        x = np.arange(len(phases_names))
        width = 0.18

        fig, ax = plt.subplots(figsize=(12, 5))
        for i, m in enumerate(models_names):
            vals = results_df[m].values
            ax.bar(x + i*width, vals, width=width, label=m)

        ax.set_xticks(x + width*(len(models_names)-1)/2)
        ax.set_xticklabels(phases_names, rotation=30, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("Phase-wise Accuracy Comparison")
        ax.legend()
        st.pyplot(fig)

        # Reports
        with st.expander("Show detailed classification reports"):
            for model_name, model in models.items():
                st.write(f"### {model_name}")
                for phase_name, X_vec in phases:
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_vec, y, test_size=0.2, random_state=42, stratify=stratify_flag
                        )
                    except Exception:
                        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

                    try:
                        need_dense = isinstance(model, DecisionTreeClassifier)
                        X_train_in = X_train.toarray() if need_dense and hasattr(X_train, "toarray") else X_train
                        X_test_in = X_test.toarray() if need_dense and hasattr(X_test, "toarray") else X_test

                        model.fit(X_train_in, y_train)
                        preds = model.predict(X_test_in)
                        report = classification_report(y_test, preds, zero_division=0, output_dict=True)
                        rpt_df = pd.DataFrame(report).transpose()
                        st.write(f"**Phase:** {phase_name}")
                        st.dataframe(rpt_df)
                    except Exception as e:
                        st.warning(f"Could not produce report for {model_name} on {phase_name}: {e}")

        # Download results
        csv = results_df.reset_index().to_csv(index=False)
        st.download_button("Download results CSV", data=csv, file_name="phase_model_comparison.csv", mime="text/csv")

else:
    st.info("Upload a CSV to get started. Example: a CSV with 'Statement' and 'Label'.")
