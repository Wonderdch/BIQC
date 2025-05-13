import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

# 数据集来源：Skin Cancer: Malignant vs. Benign
# 数据集地址：https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign

class SkinCancerDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.images = os.listdir(root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.images[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        # 根据文件夹名称设置标签
        label = -1 if 'benign' in self.root else 1
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


def generate_skin_cancer(preprocessing, height=None):
    if preprocessing != "keep_high":
        raise ValueError("只支持 'keep_high' 预处理方法")

    # 定义图像预处理流程
    transform = transforms.Compose([
        transforms.Resize((height, height)),
        transforms.Grayscale(),  # 转换为灰度图像
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 对单通道进行标准化
    ])

    # 获取数据集根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    dataset_root = os.path.join(project_root, "datasets", "skin_cancer_original")

    # 加载训练数据集
    train_benign = SkinCancerDataset(root=os.path.join(dataset_root, "train", "benign"), transform=transform)
    train_malignant = SkinCancerDataset(root=os.path.join(dataset_root, "train", "malignant"), transform=transform)
    train_dataset = ConcatDataset([train_benign, train_malignant])

    # 加载测试数据集
    test_benign = SkinCancerDataset(root=os.path.join(dataset_root, "test", "benign"), transform=transform)
    test_malignant = SkinCancerDataset(root=os.path.join(dataset_root, "test", "malignant"), transform=transform)
    test_dataset = ConcatDataset([test_benign, test_malignant])

    # 提取训练数据和标签
    X_train = torch.stack([img for img, _ in train_dataset])
    y_train = torch.tensor([label for _, label in train_dataset])

    # 提取测试数据和标签
    X_test = torch.stack([img for img, _ in test_dataset])
    y_test = torch.tensor([label for _, label in test_dataset])

    # 将图像转换为二维数组，每行代表一个样本
    X_train_flat = X_train.view(X_train.size(0), -1)
    X_test_flat = X_test.view(X_test.size(0), -1)

    return X_train_flat, X_test_flat, y_train, y_test