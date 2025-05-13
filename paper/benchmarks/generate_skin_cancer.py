import os
import torch
import numpy as np
from qml_benchmarks.data.skin_cancer import generate_skin_cancer

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)

# 生成皮肤癌 保留高频分量基准数据集
os.makedirs("./datasets/skin_cancer_kh", exist_ok=True)

for height in [224, 96]:
    X_train, X_test, y_train, y_test = generate_skin_cancer(
        preprocessing="keep_high", height=height
    )

    name_train = f"./datasets/skin_cancer_kh/skin_cancer_kh_{height}x{height}_train.csv"
    data_train = np.c_[X_train.numpy(), y_train.numpy()]
    np.savetxt(name_train, data_train, delimiter=",")
    name_test = f"./datasets/skin_cancer_kh/skin_cancer_kh_{height}x{height}_test.csv"
    data_test = np.c_[X_test.numpy(), y_test.numpy()]
    np.savetxt(name_test, data_test, delimiter=",")

print("皮肤癌 keep_high 数据生成完成。")
