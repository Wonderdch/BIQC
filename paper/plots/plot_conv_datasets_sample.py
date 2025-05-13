import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_dataset_samples():
    # 设置Times New Roman字体
    plt.rcParams['font.family'] = 'Times New Roman'

    # 定义数据集顺序和它们的显示名称
    datasets = [
        ('MNIST-8x8', 'MNIST 8×8'),
        ('MNIST-32x32', 'MNIST 32×32'),
        ('STL-96x96', 'STL 96×96'),
        ('skin-cancer-224x224', 'Skin Cancer 224×224'),
        ('DTD-224x224', 'DTD 224×224')
    ]

    # 创建图表
    fig = plt.figure(figsize=(8, 10))

    # 遍历每个数据集
    for row, (dataset_prefix, dataset_name) in enumerate(datasets):
        # 获取该数据集的所有图片和对应的标签
        images = []
        labels = []  # 创建标签列表而不是单个标签

        for file in sorted(os.listdir('figures/single')):
            if file.startswith(dataset_prefix):
                # 从文件名中提取标签
                # 例如：从 'dtd-224x224_chequered_1.png' 提取 'chequered'
                parts = file.split('_')
                if len(parts) >= 2:
                    label = parts[1].capitalize()  # 首字母大写
                    labels.append(label)

                img_path = os.path.join('figures/single', file)
                img = Image.open(img_path)
                img_array = np.array(img)
                images.append(img_array)

        # 确保找到了4张图片
        if len(images) != 4:
            print(f"Warning: Expected 4 images for {dataset_prefix}, found {len(images)}")
            continue

        # 在当前行绘制4张图片
        for col, (img, label) in enumerate(zip(images, labels)):
            ax = plt.subplot(5, 4, row * 4 + col + 1)

            # 如果是灰度图，使用gray colormap
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)

            plt.axis('off')

            # 只在每行的第一张图左侧添加数据集名称
            if col == 0:
                ax.text(-0.1, 0.5, dataset_name,
                        rotation=90,
                        transform=ax.transAxes,
                        verticalalignment='center',
                        horizontalalignment='center',
                        fontfamily='Times New Roman',
                        fontsize=12)

            # 在每张图上方添加其对应的标签
            ax.text(0.5, 1.03, label,
                    transform=ax.transAxes,
                    horizontalalignment='center',
                    fontfamily='Times New Roman',
                    fontsize=12)

    # 调整布局
    plt.tight_layout()

    # 自定义调整以减小列间距，增加左侧边距
    plt.subplots_adjust(
        left=0.03,  # 左边距
        right=0.98,  # 右边距
        wspace=0.01,  # 列间距（减小这个值）
        hspace=0.2,  # 行间距
        top=0.98  # 顶部边距（为标签留出空间）
    )

    plt.show()
    # 保存图像
    plt.savefig('figures/single/dataset_samples.png', dpi=300, bbox_inches='tight')

    # 保存 PDF 版本
    plt.savefig('figures/single/dataset_samples.pdf', dpi=300, bbox_inches='tight')

    plt.close()
    print("数据集样本图像已保存为 dataset_samples.png")


if __name__ == "__main__":
    plot_dataset_samples()
