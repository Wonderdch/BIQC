import os
import torch
import numpy as np
from qml_benchmarks.data.stl import generate_stl

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)

# 定义类别：飞机(0)和鸟(2)
classA = 0  # 飞机
classB = 1  # 鸟

# 生成STL-10 保留高频分量基准数据集
os.makedirs("./datasets/stl_kh", exist_ok=True)

for height in [32, 64, 96]:
    X_train, X_test, y_train, y_test = generate_stl(
        classA, classB, preprocessing="keep_high", height=height
    )

    name_train = f"./datasets/stl_kh/stl_kh_{classA}-{classB}_{height}x{height}_train.csv"
    data_train = np.c_[X_train.numpy(), y_train.numpy()]
    np.savetxt(name_train, data_train, delimiter=",")
    name_test = f"./datasets/stl_kh/stl_kh_{classA}-{classB}_{height}x{height}_test.csv"
    data_test = np.c_[X_test.numpy(), y_test.numpy()]
    np.savetxt(name_test, data_test, delimiter=",")

print("STL-10 keep_high 数据生成完成。")
