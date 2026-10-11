# HW-1
自然语言处理作业一

# NLP Homework 1: 新闻文本分类
本仓库为自然语言处理课程第一次作业，实现并比较不同文本表示方法在新闻文本分类任务中的性能，包括 **Bag of Words**、**Word Embedding** 和 **BERT Fine-tuning**。
主要使用 **NYT** 新闻数据集进行文本分类，并使用 **AG News** 数据集训练 Word2Vec 词向量。

---

## 项目结构

```
HW-1/
├── code/                              # 全部代码
│   ├── T1.py                          # Task 1: Bag of Words
│   ├── T2.py                          # Task 2: Word Embeddings
│   ├── T3.py                          # Task 3: BERT Fine-tuning
│   └── run.sh                         # 一键运行脚本
├── data/                              # 数据集与词向量
│   ├── nyt.csv                        # NYT 数据集
│   ├── ag.csv                         # AG News 数据集
│   └── glove.6B/
│       └── glove.6B.100d.txt          # GloVe 100 维词向量
├── models/
│   └── bert-base-uncased/             # BERT 预训练模型
├── report.pdf                         # 实验报告
└── README.md
```

---

## 环境配置

推荐 Python 3.10。

```bash
conda create -n nlp_hw1 python=3.10 -y
conda activate nlp_hw1

pip install numpy pandas scipy scikit-learn gensim nltk
pip install torch transformers datasets accelerate
```

BERT 微调建议使用 GPU。

---

## 数据集

- **NYT**：主分类数据集，含 `text`、`label` 两列，共 11,519 条，3 类（business / politics / sports）。
- **AG News**：仅用于 Task 2 训练 Word2Vec。
- **GloVe 100d**：下载 https://nlp.stanford.edu/data/glove.6B.zip ，取 `glove.6B.100d.txt` 放到 `data/glove.6B/`。

数据划分：Training / Validation / Test = 80% / 10% / 10%，分层抽样，`random_state=42`。

---

## 运行方式

```bash
cd code

python T1.py     # Task 1: Bag of Words
python T2.py     # Task 2: Word Embeddings
python T3.py     # Task 3: BERT Fine-tuning
```

或一键运行：

```bash
bash run.sh
```

---

## 实验结果（Test Set）

| 方法 | Accuracy | Macro-F1 |
|---|---|---|
| Binary Bag of Words | 0.9902 | 0.9761 |
| Word Frequency | 0.9893 | 0.9739 |
| GloVe 6B 100d | 0.9841 | 0.9621 |
| Word2Vec (AG News) | 0.9780 | 0.9479 |
| Word2Vec (NYT train) | 0.9815 | 0.9529 |
| BERT-base-uncased | 0.9832 | 0.9644 |

---

## 主要结论

1. **词袋模型在该任务上最强**：NYT 三分类本质是关键词驱动任务，Binary BoW 取得最佳 Accuracy 与 Macro-F1。
2. **词向量方法中领域匹配至关重要**：GloVe > Word2Vec(NYT) > Word2Vec(AG)，领域不匹配会显著拉低性能。
3. **BERT 类别均衡性最优**：Accuracy 不占优，但 Macro-F1 在词向量类方法中最高，对小类别最公平。

---

## 实验报告

详细方法原理、实验设置与结果分析见 **`report.pdf`**。
```

直接保存为 `README.md` 上传即可。
