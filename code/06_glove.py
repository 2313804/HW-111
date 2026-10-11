import nltk
nltk.data.path.append("D:/nltk_data")

import os
import numpy as np
import pandas as pd
from nltk import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ---------- 1. 读数据 ----------
train = pd.read_csv("data/nyt_train.csv")
valid = pd.read_csv("data/nyt_valid.csv")
test  = pd.read_csv("data/nyt_test.csv")

# ---------- 2. 加载 GloVe ----------
GLOVE_PATH = "data/glove/glove.6B.100d.txt"
assert os.path.exists(GLOVE_PATH), f"GloVe 文件不存在: {GLOVE_PATH}"

print("Loading GloVe ...")
embeddings = {}
with open(GLOVE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip().split(" ")
        word = parts[0]
        vec = np.asarray(parts[1:], dtype="float32")
        embeddings[word] = vec

print("GloVe vocab size:", len(embeddings))
DIM = 100

# ---------- 3. 文档向量 = 词向量平均 ----------
def tokenize(text):
    return word_tokenize(str(text).lower())

def doc_vector(tokens):
    vecs = [embeddings[w] for w in tokens if w in embeddings]
    if len(vecs) == 0:
        return np.zeros(DIM)
    return np.mean(vecs, axis=0)

print("Building document vectors ...")
X_train = np.vstack([doc_vector(tokenize(t)) for t in train["text"]])
X_valid = np.vstack([doc_vector(tokenize(t)) for t in valid["text"]])
X_test  = np.vstack([doc_vector(tokenize(t)) for t in test["text"]])

y_train = train["label"]
y_valid = valid["label"]
y_test  = test["label"]

# ---------- 4. Logistic Regression ----------
print("Training Logistic Regression ...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

valid_pred = clf.predict(X_valid)
print("Valid Accuracy:", accuracy_score(y_valid, valid_pred))
print("Valid Macro-F1:", f1_score(y_valid, valid_pred, average="macro"))

test_pred = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print("Test Macro-F1:", f1_score(y_test, test_pred, average="macro"))