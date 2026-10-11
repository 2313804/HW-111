import pandas as pd
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# 读数据
train = pd.read_csv("data/nyt_train.csv")
valid = pd.read_csv("data/nyt_valid.csv")
test  = pd.read_csv("data/nyt_test.csv")

# 分词函数
def tokenize(text):
    return word_tokenize(text.lower())

# 对三个集合分词
train_tokens = train["text"].apply(tokenize)
valid_tokens = valid["text"].apply(tokenize)
test_tokens  = test["text"].apply(tokenize)

# 把 token 列表拼回字符串，CountVectorizer 需要字符串输入
train_text = train_tokens.apply(lambda x: " ".join(x))
valid_text = valid_tokens.apply(lambda x: " ".join(x))
test_text  = test_tokens.apply(lambda x: " ".join(x))

# Binary Bag of Words
vectorizer = CountVectorizer(binary=True)
X_train = vectorizer.fit_transform(train_text)
X_valid = vectorizer.transform(valid_text)
X_test  = vectorizer.transform(test_text)

print("Vocab size:", len(vectorizer.vocabulary_))
print("X_train shape:", X_train.shape)

# 标签
y_train = train["label"]
y_valid = valid["label"]
y_test  = test["label"]

# Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 在 validation 上看一下
valid_pred = clf.predict(X_valid)
print("Valid Accuracy:", accuracy_score(y_valid, valid_pred))
print("Valid Macro-F1:", f1_score(y_valid, valid_pred, average="macro"))

# 在 test 上报告最终结果
test_pred = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print("Test Macro-F1:", f1_score(y_test, test_pred, average="macro"))