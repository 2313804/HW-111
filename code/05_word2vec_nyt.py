import nltk
nltk.data.path.append("D:/nltk_data")

import pandas as pd
import numpy as np
from nltk import word_tokenize
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# ---------- 1. 读数据 ----------
train = pd.read_csv("data/nyt_train.csv")
valid = pd.read_csv("data/nyt_valid.csv")
test  = pd.read_csv("data/nyt_test.csv")

# ---------- 2. 分词 ----------
def tokenize(text):
    return word_tokenize(str(text).lower())

print("Tokenizing NYT train ...")
nyt_tokens = [tokenize(t) for t in train["text"]]

# ---------- 3. 训练 Word2Vec ----------
print("Training Word2Vec on NYT ...")
w2v = Word2Vec(
    sentences=nyt_tokens,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=5,
    seed=42
)

# ---------- 4. 文档向量 = 词向量平均 ----------
def doc_vector(tokens, model):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if len(vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

print("Building NYT document vectors ...")
X_train = np.vstack([doc_vector(tokenize(t), w2v) for t in train["text"]])
X_valid = np.vstack([doc_vector(tokenize(t), w2v) for t in valid["text"]])
X_test  = np.vstack([doc_vector(tokenize(t), w2v) for t in test["text"]])

y_train = train["label"]
y_valid = valid["label"]
y_test  = test["label"]

# ---------- 5. Logistic Regression ----------
print("Training Logistic Regression ...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

valid_pred = clf.predict(X_valid)
print("Valid Accuracy:", accuracy_score(y_valid, valid_pred))
print("Valid Macro-F1:", f1_score(y_valid, valid_pred, average="macro"))

test_pred = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print("Test Macro-F1:", f1_score(y_test, test_pred, average="macro"))

w2v.save("models/w2v_nyt.model")
print("Saved models/w2v_nyt.model")