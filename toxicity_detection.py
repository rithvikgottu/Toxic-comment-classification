import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# Loading data
df = pd.read_csv("train.csv")

# Looking at first couple rows
print(df.head())

# Checking columns
print("Columns: " + df.columns)

# Check how many examples are labeled for each toxic category
print(df.iloc[:, 2:].sum())

# Preprocess the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = text.replace("\n", " ")
    text = text.strip()
    return text

# Clean the data
df["comment_text"] = df["comment_text"].apply(clean_text)
print(df["comment_text"].head())

# Features and Labels
X = df["comment_text"]
y = df.iloc[:, 2:]

# Splitting data into 70% training and 30% temp (validation and test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Split the temp into the validation and test
X_val, X_test, y_val, y_test =  train_test_split(
    X_temp, y_temp, test_size = 0.5, random_state=42)

# Print sizes
print("Train_size: ", X_train.shape[0])
print("Validation size: ", X_val.shape[0])
print("Test size: ", X_test.shape[0])

# Initialize TF-IDF Vector
vectorizer = TfidfVectorizer(max_features=50000, stop_words="english")

# Fit on our training data and transform
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform validation and test data
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF feature shape: ", X_train_tfidf.shape)

# Store the f1 scores for the validation
scores = {}

# Train 1 logistic Regression model for each toxicity label 
for label in y_train.columns:
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_tfidf, y_train[label])
    preds = model.predict(X_val_tfidf)
    scores[label] = f1_score(y_val[label], preds)

print("Validation F1 scores:", scores)

# Testing
test_scores = {}
models = {}
for label in y_train.columns:
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_tfidf, y_train[label])
    preds = model.predict(X_test_tfidf)
    test_scores[label] = f1_score(y_test[label], preds)
    models[label] = model

print("Test F1 scores:", test_scores)

# Visualizing results
labels = list(test_scores.keys())
test_scores_values = list(test_scores.values())
plt.bar(labels, test_scores_values)
plt.ylabel("F1 Score")
plt.title("F1 scores for test set")
plt.show()

# Doing Sample Predictions

print("\nSample Predictions on Data\n")

for i in range(5):  #5 examples
    comment = X_test.iloc[i]
    true_labels = y_test.iloc[i].to_dict()
    comment_tfidf = vectorizer.transform([comment])
    pred_labels = {label: models[label].predict(comment_tfidf)[0] for label in y_train.columns}

    print(f"Comment: {comment}")
    print(f"True Labels: {true_labels}")
    print(f"Predicted Labels: {pred_labels}")
    print("-" * 50)