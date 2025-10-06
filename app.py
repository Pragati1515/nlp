# =========================================
# Imports & Downloads
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# NLTK downloads
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# Initialize lemmatizer & stopwords
lemmatizer = WordNetLemmatizer()
custom_stopwords = set(
    w for w in stopwords.words("english")
    if w not in ["not", "no", "nor", "against", "is", "are", "be", "been"]
)

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# =========================================
# Preprocessing Functions
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
        return " ".join(pos_tags) if pos_tags else str(text)
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
        return f"{len(sentences)} {' '.join(first_words)}" if sentences else "0"
    except:
        return "0"

def pragmatic_features(text):
    try:
        tokens = []
        for w in pragmatic_words:
            count = str(text).lower().count(w)
            tokens.extend([w] * count)
        return " ".join(tokens) if tokens else str(text)
    except:
        return str(text)

# =========================================
# Safe Phase-wise Vectorization
# =========================================
def vectorize_phase(train_texts, test_texts, phase_name, max_features=5000):
    train_texts = [str(t) for t in train_texts]
    test_texts  = [str(t) for t in test_texts]

    # Choose vectorizer
    if phase_name in ["Lexical & Morphological", "Syntactic"]:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b")
    else:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=3000)

    try:
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        if X_train.shape[1] == 0:
            raise ValueError("Empty vocabulary")
        return vectorizer, X_train, X_test
    except:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=1000)
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        return vectorizer, X_train, X_test

# =========================================
# Train & Evaluate
# =========================================
def train_and_eval(model, X_train, X_test, y_train, y_test):
    try:
        # Convert to dense if needed
        if isinstance(model, DecisionTreeClassifier):
            X_train_in = X_train.toarray() if hasattr(X_train, "toarray") else X_train
            X_test_in = X_test.toarray() if hasattr(X_test, "toarray") else X_test
        else:
            X_train_in, X_test_in = X_train, X_test

        model.fit(X_train_in, y_train)
        preds = model.predict(X_test_in)
        acc = accuracy_score(y_test, preds)
        return acc, classification_report(y_test, preds, zero_division=0, output_dict=True)
    except:
        return 0.0, {}

# =========================================
# Streamlit App
# =========================================
st.title("📰 Fake News Detection: Phase-wise NLP")

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

        # Apply preprocessing phases
        train_phases = {
            "Lexical & Morphological": X_train.apply(lexical_preprocess),
            "Syntactic": X_train.apply(syntactic_features),
            "Semantic": X_train.apply(semantic_features),
            "Discourse": X_train.apply(discourse_features),
            "Pragmatic": X_train.apply(pragmatic_features),
        }
        test_phases = {
            "Lexical & Morphological": X_test.apply(lexical_preprocess),
            "Syntactic": X_test.apply(syntactic_features),
            "Semantic": X_test.apply(semantic_features),
            "Discourse": X_test.apply(discourse_features),
            "Pragmatic": X_test.apply(pragmatic_features),
        }

        # Models
        models = {
            "Naive Bayes": MultinomialNB(),
            "SVM": SVC(kernel="linear", probability=True),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(max_depth=20),
        }

        results = []
        reports = {}

        # Train & evaluate
        for model_name, model in models.items():
            for phase_name in train_phases.keys():
                _, Xtr, Xte = vectorize_phase(train_phases[phase_name], test_phases[phase_name], phase_name)
                acc, rpt = train_and_eval(model, Xtr, Xte, y_train, y_test)
                results.append({"Phase": phase_name, "Model": model_name, "Accuracy": acc})
                reports[(model_name, phase_name)] = rpt

        # Results Table
        results_df = pd.DataFrame(results)
        pivot_df = results_df.pivot(index="Phase", columns="Model", values="Accuracy")
        st.subheader("📊 Accuracy Comparison Table")
        st.dataframe(pivot_df.style.format("{:.4f}"))

        # Grouped Bar Chart
        st.subheader("📈 Phase-wise Accuracy by Model")
        fig, ax = plt.subplots(figsize=(10,6))
        x = np.arange(len(pivot_df.index))
        width = 0.18
        for i, model in enumerate(pivot_df.columns):
            ax.bar(x + i*width, pivot_df[model], width=width, label=model)
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(pivot_df.index, rotation=25)
        ax.set_ylim(0,1)
        ax.legend()
        st.pyplot(fig)

        # Predict New Text (Optional)
        st.subheader("📝 Predict New Text")
        new_text = st.text_area("Enter text to predict")
        if st.button("Predict"):
            if new_text.strip():
                pred_results = []
                for model_name, model in models.items():
                    for phase_name in train_phases.keys():
                        _, Xtr_dummy, _ = vectorize_phase(train_phases[phase_name], [new_text], phase_name)
                        pred = model.predict(Xtr_dummy)
                        pred_results.append(f"{model_name} ({phase_name}): {pred[0]}")
                st.write(pred_results)
            else:
                st.warning("Please enter some text!")

    except Exception as e:
        st.error(f"Error: {e}")
