# Sentiment Analysis on Twitter Data

## Project Overview
This project uses Machine Learning and Natural Language Processing (NLP) techniques to classify Twitter sentiments as Positive or Negative.

The model analyzes tweet text data and predicts sentiment using:
- TF-IDF Vectorization
- Logistic Regression

This project demonstrates:
- Text preprocessing
- Feature extraction
- Sentiment classification
- Model evaluation


# Technologies Used
- Python
- Pandas
- NumPy
- NLTK
- Scikit-Learn
- TF-IDF Vectorization
- Logistic Regression

# Machine Learning Workflow
1. Data Collection
2. Data Cleaning
3. Text Preprocessing
4. Stopword Removal
5. Feature Extraction using TF-IDF
6. Train-Test Split
7. Model Training
8. Sentiment Prediction
9. Model Evaluation


# Dataset
Dataset Used:
Sentiment140 Twitter Dataset
Dataset Link:
https://www.kaggle.com/datasets/kazanova/sentiment140



# Project Structure
```text
Sentiment_Analysis_Project/
│
├── Output/
│     └── Accuracy_and_Classification_Report.png
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```


# How to Run the Project
## Step 1: Clone Repository
```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

## Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 3: Run the Project
```bash
python app.py
```


# Machine Learning Model
Algorithm Used:
- Logistic Regression

Feature Extraction Technique:
- TF-IDF Vectorization



# Model Performance
## Accuracy

```text
Accuracy: 76.85%
```

# Classification Report
| Class | Precision | Recall | F1-Score |
|------|------|------|------|
| Negative Sentiment (0) | 0.78 | 0.74 | 0.76 |
| Positive Sentiment (1) | 0.76 | 0.79 | 0.77 |



# Interpretation of Results
- The model achieved approximately **76.85% accuracy** on unseen Twitter data.
- Precision and recall values indicate balanced model performance for both positive and negative sentiment classes.
- The model performs slightly better at identifying positive sentiments.
- TF-IDF vectorization successfully converted textual tweet data into numerical features for machine learning.
- Logistic Regression provided efficient and interpretable classification performance for NLP tasks.

Since Twitter data contains:
- slang,
- abbreviations,
- emojis,
- spelling variations,
- noisy text,

an accuracy of around 77% is considered good for a classical machine learning approach.

# Output Screenshot
## Accuracy and Classification Report
![Model Output](Output/Accuracy_Report.png)

# Future Improvements
- Multi-class Sentiment Analysis
- Deep Learning Models (LSTM, GRU)
- BERT Transformer Models
- Real-time Twitter API Integration
- Emotion Detection

# Applications
- Social Media Monitoring
- Brand Reputation Analysis
- Customer Feedback Analysis
- Public Opinion Mining
- Market Research