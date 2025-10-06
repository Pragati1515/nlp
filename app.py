# ====================================
# Improved Preprocessing Functions
# ====================================
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

nltk.download("averaged_perceptron_tagger")

# Light stopwords: exclude negations and auxiliaries
custom_stopwords = set(w for w in stopwords.words("english") if w not in ["not", "no", "nor", "was", "is", "are", "be", "been"])

def lexical_preprocess(text):
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)
    cleaned = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in custom_stopwords
        and w not in string.punctuation
        and len(w) > 1
    ]
    return " ".join(cleaned)

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
def vectorize_phase(train_texts, test_texts, phase_name):
    """
    Phase-aware vectorizer: uses suitable TF-IDF config for each feature type.
    """
    if phase_name in ["Lexical & Morphological", "Syntactic"]:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
        )
    elif phase_name in ["Semantic", "Discourse", "Pragmatic"]:
        # smaller feature set, use char n-grams for robustness
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=3000
        )
    else:
        vectorizer = TfidfVectorizer(max_features=5000)

    vectorizer.fit(train_texts)
    return vectorizer.transform(train_texts), vectorizer.transform(test_texts)

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
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        text_col = st.selectbox("Select TEXT column", df.columns)
        target_col = st.selectbox("Select TARGET column", df.columns)

        data = df[[text_col, target_col]].dropna().copy()
        data.columns = ["text", "target"]

        # Label encode target if needed
        if data["target"].dtype == object:
            data["target"] = LabelEncoder().fit_transform(data["target"].astype(str))

        X_train, X_test, y_train, y_test = train_test_split(
            data["text"].astype(str), data["target"],
            test_size=0.2, random_state=42, stratify=data["target"]
        )

        # Phase transformations
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

        for model_name, model in models.items():
            for phase_name in train_phases.keys():
                Xtr, Xte = vectorize_phase(train_phases[phase_name], test_phases[phase_name], phase_name)
                acc, rpt = train_and_eval(model, Xtr, Xte, y_train, y_test)
                results.append({"Phase": phase_name, "Model": model_name, "Accuracy": acc})
                reports[(model_name, phase_name)] = rpt

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

    except Exception as e:
        st.error(f"Error: {e}")
