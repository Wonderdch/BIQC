import os
import torch
import numpy as np
from qml_benchmarks.data.dtd import generate_dtd

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)

# 生成 DTD 保留高频分量基准数据集
os.makedirs("./datasets/dtd_kh", exist_ok=True)

for height in [224, 96, 32]:
    # 数据量太小了，就没划分 train 和 test
    X_train, y_train = generate_dtd(
        preprocessing="keep_high", height=height
    )

    name_train = f"./datasets/dtd_kh/dtd_kh_{height}x{height}_train.csv"
    data_train = np.c_[X_train.numpy(), y_train.numpy()]
    np.savetxt(name_train, data_train, delimiter=",")

print("DTD keep_high 数据生成完成。")
