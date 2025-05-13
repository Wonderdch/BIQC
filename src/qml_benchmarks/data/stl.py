import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# 数据集来源：STL-10
# 数据集地址：https://www.kaggle.com/datasets/jessicali9530/stl10/

def generate_stl(classA, classB, preprocessing, height=None):
    if preprocessing != "keep_high":
        raise ValueError("只支持 'keep_high' 预处理方法")

    # 定义一个函数来筛选所需的类别
    def filter_classes(dataset, classA, classB):
        mask = torch.tensor(dataset.labels) == classA
        mask |= torch.tensor(dataset.labels) == classB
        dataset.data = dataset.data[mask]
        dataset.labels = torch.tensor(dataset.labels)[mask]
        return dataset

    # 下载并立即筛选数据
    stl_train = torchvision.datasets.STL10(root="./datasets/stl_original", split="train", download=True)
    stl_train = filter_classes(stl_train, classA, classB)

    stl_test = torchvision.datasets.STL10(root="./datasets/stl_original", split="test", download=True)
    stl_test = filter_classes(stl_test, classA, classB)

    # 转换标签为 -1 和 1
    y_train = torch.where(torch.tensor(stl_train.labels) == classA, -1, 1)
    y_test = torch.where(torch.tensor(stl_test.labels) == classA, -1, 1)

    # 定义图像预处理流程
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),  # 转换为灰度图像
        transforms.Resize(height),
        transforms.CenterCrop(height),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 对单通道进行标准化
    ])

    # 应用预处理
    X_train = torch.stack([transform(img) for img in stl_train.data.transpose(0, 2, 3, 1)])
    X_test = torch.stack([transform(img) for img in stl_test.data.transpose(0, 2, 3, 1)])

    # 将图像转换为二维数组，每行代表一个样本
    X_train_flat = X_train.view(X_train.size(0), -1)
    X_test_flat = X_test.view(X_test.size(0), -1)

    return X_train_flat, X_test_flat, y_train, y_test
