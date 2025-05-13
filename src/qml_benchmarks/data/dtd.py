import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

# 数据集来源：Describable Textures Dataset (DTD): chequered vs. striped
# 数据集地址：https://www.kaggle.com/datasets/jmexpert/describable-textures-dataset-dtd/

class DTDDataset(Dataset):
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
        label = -1 if 'chequered' in self.root else 1  # dotted的标签为1
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

def generate_dtd(preprocessing, height=None):
    if preprocessing != "keep_high":
        raise ValueError("只支持 'keep_high' 预处理方法")

    # 定义图像预处理流程，包含数据增强
    transform = transforms.Compose([
        transforms.RandomCrop((height, height)), # 随机裁剪
        transforms.RandomHorizontalFlip(),       # 随机水平翻转
        transforms.RandomVerticalFlip(),         # 随机垂直翻转
        transforms.RandomRotation(10),           # 随机旋转
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # 随机调整亮度、对比度等
        transforms.Grayscale(),                  # 转换为灰度图像
        transforms.ToTensor(),                   # 转换为张量
        transforms.Normalize((0.5,), (0.5,))     # 对单通道进行标准化
    ])

    # 获取数据集根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    dataset_root = os.path.join(project_root, "datasets", "dtd_original")

    # 定义一个新的数据集类，包含数据增强
    class AugmentedDTDDataset(Dataset):
        def __init__(self, root, transform=None, augment_factor=5):
            self.root = root
            self.transform = transform
            self.augment_factor = augment_factor
            self.images = os.listdir(root)

        def __len__(self):
            return len(self.images) * self.augment_factor

        def __getitem__(self, idx):
            img_idx = idx % len(self.images)
            img_name = os.path.join(self.root, self.images[img_idx])
            image = Image.open(img_name)
            
            if self.transform:
                image = self.transform(image)
            
            # 根据文件夹名称设置标签
            label = -1 if 'chequered' in self.root else 1  # dotted的标签为1
            
            return image, label
        
    # 加载训练数据集
    train_chequered = AugmentedDTDDataset(root=os.path.join(dataset_root, "train", "chequered"), transform=transform)
    train_dotted = AugmentedDTDDataset(root=os.path.join(dataset_root, "train", "dotted"), transform=transform)
    train_dataset = ConcatDataset([train_chequered, train_dotted])

    # 提取训练数据和标签
    X_train = torch.stack([img for img, _ in train_dataset])
    y_train = torch.tensor([label for _, label in train_dataset])

    # 将图像转换为二维数组，每行代表一个样本
    X_train_flat = X_train.view(X_train.size(0), -1)

    return X_train_flat, y_train
