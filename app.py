import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#Download stopwords
nltk.download('stopwords')

#Load dataset
data = pd.read_csv(
    "dataset/tweets.csv",
    encoding = 'latin-1',
    header=None
)

#Select the required columns
data = data[[0,5]]

#Rename the columns
data.columns = ['sentiment', 'tweet']

#Convert sentiment
data['sentiment'] = data['sentiment'].replace(4,1)

#Stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Apply cleaning
data['tweet'] = data['tweet'].apply(clean_text)

# Features and labels
X = data['tweet']
y = data['sentiment']

# Convert text to vectors
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Custom prediction
sample = ["I love bioinformatics and AI"]

sample = vectorizer.transform(sample)

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\nPositive Sentiment")
else:
    print("\nNegative Sentiment")