import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ---------- 1. 读数据 ----------
train = pd.read_csv("data/nyt_train.csv")
valid = pd.read_csv("data/nyt_valid.csv")
test  = pd.read_csv("data/nyt_test.csv")

le = LabelEncoder()
y_train = le.fit_transform(train["label"])
y_valid = le.transform(valid["label"])
y_test  = le.transform(test["label"])
num_labels = len(le.classes_)
print("Labels:", list(le.classes_))

# ---------- 2. Tokenizer ----------
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class NYTDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_ds = NYTDataset(train["text"], y_train)
valid_ds = NYTDataset(valid["text"], y_valid)
test_ds  = NYTDataset(test["text"], y_test)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=32)
test_loader  = DataLoader(test_ds,  batch_size=32)

# ---------- 3. 模型 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels
).to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# ---------- 4. 训练 3 epochs ----------
def evaluate(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())
    return np.array(labels), np.array(preds)

for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")

    y_v, p_v = evaluate(valid_loader)
    print(f"  Valid Acc: {accuracy_score(y_v, p_v):.4f}  Macro-F1: {f1_score(y_v, p_v, average='macro'):.4f}")

# ---------- 5. Test ----------
y_t, p_t = evaluate(test_loader)
print("Test Accuracy:", accuracy_score(y_t, p_t))
print("Test Macro-F1:", f1_score(y_t, p_t, average="macro"))

model.save_pretrained("models/bert_nyt")
tokenizer.save_pretrained("models/bert_nyt")
print("Saved models/bert_nyt")