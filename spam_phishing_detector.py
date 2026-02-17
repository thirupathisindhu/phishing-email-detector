import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# -------------------------
# Load Dataset
# -------------------------
data = pd.read_csv("emails.csv")

# -------------------------
# Text Preprocessing
# -------------------------
def preprocess(text):
    tokens = word_tokenize(text.lower())
    
    words = [
        word for word in tokens
        if word.isalpha()
        and word not in stopwords.words("english")
    ]
    
    return " ".join(words)

data["clean"] = data["text"].apply(preprocess)

# -------------------------
# Feature Extraction
# -------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["clean"])
y = data["label"]

# -------------------------
# Train Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Train Model
# -------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
pred = model.predict(X_test)
print(classification_report(y_test, pred))

# -------------------------
# Test with New Email
# -------------------------
def predict_email(email):
    email_clean = preprocess(email)
    vec = vectorizer.transform([email_clean])
    result = model.predict(vec)[0]
    return result

print("\nTest Prediction:")
print(predict_email("Verify your account immediately to avoid suspension"))
