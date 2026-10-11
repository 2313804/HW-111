import nltk
nltk.data.path.append("D:/nltk_data")   # 保险起见加上

import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

train = pd.read_csv("data/nyt_train.csv")
valid = pd.read_csv("data/nyt_valid.csv")
test  = pd.read_csv("data/nyt_test.csv")

def tokenize(text):
    return word_tokenize(text.lower())

train_text = train["text"].apply(lambda x: " ".join(tokenize(x)))
valid_text = valid["text"].apply(lambda x: " ".join(tokenize(x)))
test_text  = test["text"].apply(lambda x: " ".join(tokenize(x)))

# Word Frequency: binary=False
vectorizer = CountVectorizer(binary=False)
X_train = vectorizer.fit_transform(train_text)
X_valid = vectorizer.transform(valid_text)
X_test  = vectorizer.transform(test_text)

print("Vocab size:", len(vectorizer.vocabulary_))
print("X_train shape:", X_train.shape)

y_train = train["label"]
y_valid = valid["label"]
y_test  = test["label"]

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

valid_pred = clf.predict(X_valid)
print("Valid Accuracy:", accuracy_score(y_valid, valid_pred))
print("Valid Macro-F1:", f1_score(y_valid, valid_pred, average="macro"))

test_pred = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print("Test Macro-F1:", f1_score(y_test, test_pred, average="macro"))