import pandas as pd
from sklearn.model_selection import train_test_split

# 读数据
df = pd.read_csv("data/nyt.csv")

# 打乱
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 先分 train 和 temp (80% / 20%)
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

# temp 再分 valid 和 test (各 10%)
valid_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
)

# 保存
train_df.to_csv("data/nyt_train.csv", index=False)
valid_df.to_csv("data/nyt_valid.csv", index=False)
test_df.to_csv("data/nyt_test.csv", index=False)

# 打印信息
print("Train:", len(train_df))
print("Valid:", len(valid_df))
print("Test :", len(test_df))
print("\nTrain label distribution:")
print(train_df["label"].value_counts())