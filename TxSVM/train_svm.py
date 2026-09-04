# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, matthews_corrcoef


# ============================
# 参数
# ============================

train_csv = "./training_set_prott5_embedding.csv"

model_path = "./svm_prott5_model.pkl"

# ============================
# 加载训练数据
# ============================

data = pd.read_csv(train_csv, dtype={"name": "string"})

print("Training data shape:", data.shape)

# label列
y_train = data["label"].values

# embedding列
# 去掉name和label

x_train = data.iloc[:, 2:].values

print("Feature shape:", x_train.shape)

# ============================
# 建立SVM模型
# ============================

svm = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

# ============================
# 模型训练
# ============================

svm.fit(x_train, y_train)

print("SVM training finished.")

# ============================
# 保存模型
# ============================

joblib.dump(svm, model_path)

print("Model saved:", model_path)