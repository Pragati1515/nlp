# ====================================
# Improved Preprocessing Functions
# ====================================
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
nltk.download('punkt_tab')

nltk.download('stopwords')
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# =========================================
# 1️⃣ Basic Libraries
# =========================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string

# =========================================
# 2️⃣ NLTK for NLP
# =========================================
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Download NLTK data (quietly, no popups)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# Initialize lemmatizer and custom stopwords
lemmatizer = WordNetLemmatizer()
custom_stopwords = set(
    w for w in stopwords.words("english")
    if w not in ["not", "no", "nor", "against", "is", "are", "be", "been"]
)

# Pragmatic words for features
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# =========================================
# 3️⃣ Sklearn for ML
# =========================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


nltk.download("averaged_perceptron_tagger")

# Light stopwords: exclude negations and auxiliaries
custom_stopwords = set(w for w in stopwords.words("english") if w not in ["not", "no", "nor", "was", "is", "are", "be", "been"])

def lexical_preprocess(text):
    try:
        tokens = nltk.word_tokenize(str(text).lower())
        cleaned = [
            lemmatizer.lemmatize(w)
            for w in tokens
            if w not in custom_stopwords and w not in string.punctuation and len(w) > 1
        ]
        return " ".join(cleaned) if cleaned else "empty_doc"
    except:
        return "empty_doc"


def syntactic_features(text):
    try:
        tokens = nltk.word_tokenize(text)
        tags = [pos for _, pos in nltk.pos_tag(tokens)]
        return " ".join(tags)
    except:
        return ""

def semantic_features(text):
    try:
        blob = TextBlob(text)
        return f"{blob.sentiment.polarity:.3f} {blob.sentiment.subjectivity:.3f}"
    except:
        return "0 0"

def discourse_features(text):
    try:
        sentences = nltk.sent_tokenize(text)
        starters = [s.split()[0] for s in sentences if s]
        return f"{len(sentences)} {' '.join(starters)}"
    except:
        return "0"

def pragmatic_features(text):
    text = text.lower()
    counts = [w for w in pragmatic_words if w in text]
    return " ".join(counts) if counts else "none"

# ====================================
# Phase Vectorization Function
# ====================================
def vectorize_phase(train_texts, test_texts, phase_name, max_features=5000):
    # Ensure strings
    train_texts = [str(t) if pd.notnull(t) else "empty_doc" for t in train_texts]
    test_texts  = [str(t) if pd.notnull(t) else "empty_doc" for t in test_texts]

    # Choose vectorizer based on phase
    if phase_name in ["Lexical & Morphological", "Syntactic"]:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), token_pattern=r"(?u)\b\w+\b")
    else:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=3000)

    # Try fitting
    try:
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        if X_train.shape[1] == 0:
            raise ValueError("Empty vocabulary detected")
        return vectorizer, X_train, X_test
    except Exception as e:
        # Fallback: char-level TF-IDF
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=1000)
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        return vectorizer, X_train, X_test

# ====================================
# Train & Evaluate
# ====================================
def train_and_eval(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0, output_dict=True)
    return acc, report

# ====================================
# Main Logic After File Upload
# ====================================

st.title("📰 Fake News Detection: Phase-wise NLP with Multiple Models")

# ====================================
# File Upload
# ====================================
uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded is not None:
    try:  # ⬅️ Start try here

        # 1️⃣ Read file & select columns
        df = pd.read_csv(uploaded)
        text_col = st.selectbox("Select TEXT column", df.columns)
        target_col = st.selectbox("Select TARGET column", df.columns)

        data = df[[text_col, target_col]].dropna().copy()
        data.columns = ["text", "target"]

        # Encode target if needed
        if data["target"].dtype == object:
            data["target"] = LabelEncoder().fit_transform(data["target"].astype(str))

        # 2️⃣ Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            data["text"].astype(str),
            data["target"],
            test_size=0.2,
            random_state=42,
            stratify=data["target"]
        )

        # 3️⃣ Preprocessing phases
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

        # 4️⃣ Model Training & Evaluation
        models = {
            "Naive Bayes": MultinomialNB(),
            "SVM": SVC(kernel="linear", probability=True),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(max_depth=20),
        }

        results = []
        reports = {}

        for model_name, model in models.items():
            for phase_name in train_phases.keys():
                _, Xtr, Xte = vectorize_phase(train_phases[phase_name], test_phases[phase_name], phase_name)
                acc, rpt = train_and_eval(model, Xtr, Xte, y_train, y_test)
                results.append({"Phase": phase_name, "Model": model_name, "Accuracy": acc})
                reports[(model_name, phase_name)] = rpt

        # 5️⃣ Results Visualization
        results_df = pd.DataFrame(results)
        pivot_df = results_df.pivot(index="Phase", columns="Model", values="Accuracy")
        st.subheader("📊 Accuracy Comparison Table")
        st.dataframe(pivot_df.style.format("{:.4f}"))

        # Bar chart
        st.subheader("📈 Phase-wise Accuracy by Model")
        fig, ax = plt.subplots(figsize=(10,6))
        x = np.arange(len(pivot_df.index))
        width = 0.18
        for i, model in enumerate(pivot_df.columns):
            ax.bar(x + i*width, pivot_df[model], width=width, label=model)
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(pivot_df.index, rotation=25)
        ax.legend()
        ax.set_ylim(0,1)
        st.pyplot(fig)

    except Exception as e:  # ⬅️ Catch errors here
        st.error(f"Error: {e}")
