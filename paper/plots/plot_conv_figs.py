import torch
import torch.nn as nn
import argparse
import os
from qml_benchmarks.hyperparam_search_utils import read_data
import matplotlib.pyplot as plt
import numpy as np


def plot_mnist_cg(size):
    # MNIST_CG 数据集绘制代码
    # note: you need to download the mnist_cg data
    X, y = read_data(f"datasets-for-plots/mnist_cg/mnist_pixels_3-5_{size}x{size}_train-20.csv")
    X = np.reshape(X, (X.shape[0], size, size))

    # 找到数字3和数字5的索引
    idx3_all = np.where(y == -1)[0]  # 数字3的所有索引
    idx5_all = np.where(y == 1)[0]  # 数字5的所有索引

    # 选择前两个索引
    idx3_selected = idx3_all[:2]
    idx5_selected = idx5_all[:2]

    # 创建保存目录
    os.makedirs('figures/single', exist_ok=True)

    # 保存每张图像
    for i, idx in enumerate(idx3_selected):
        plt.figure(figsize=(2, 2))
        plt.imshow(-X[idx], cmap='gray')
        plt.axis('off')
        plt.savefig(f'figures/single/mnist-cg-{size}x{size}_3_{i + 1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    for i, idx in enumerate(idx5_selected):
        plt.figure(figsize=(2, 2))
        plt.imshow(-X[idx], cmap='gray')
        plt.axis('off')
        plt.savefig(f'figures/single/mnist-cg-{size}x{size}_5_{i + 1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"已保存4张MNIST图像到 figures/single/")


def prepare_stl_data_for_plots():
    # 获取项目根目录的绝对路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # 构建输入和输出文件夹的绝对路径
    input_folder = os.path.join(project_root, "datasets", "stl_kh")
    output_folder = os.path.join(project_root, "paper", "plots", "datasets-for-plots", "stl_kh")

    # 确保输入文件夹存在
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"错误: 输入文件夹 '{input_folder}' 不存在")

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 只处理96x96的数据
    filename = "stl_kh_0-1_96x96_train.csv"
    if filename.endswith("_train.csv"):
        input_path = os.path.join(input_folder, filename)
        output_filename = filename.replace("_train.csv", "_train-20.csv")
        output_path = os.path.join(output_folder, output_filename)
        try:
            with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
                # 直接复制前 20 行
                for _ in range(20):
                    line = infile.readline()
                    if line:
                        outfile.write(line)
                    else:
                        break  # 如果文件行数少于 20，则提前结束

            print(f"已创建文件：{output_path}")
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")

    print("数据准备完成")


def plot_stl():
    # 获取项目根目录的绝对路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    size = 96  # 只处理96x96的图像
    data_folder = os.path.join(project_root, "paper", "plots", "datasets-for-plots", "stl_kh")

    # 检查是否存在用于绘图的 STL-10 数据集
    filename = os.path.join(data_folder, f"stl_kh_0-1_{size}x{size}_train-20.csv")
    if not os.path.exists(filename):
        print(f"未找到用于绘图的 STL-10 数据集: {filename}")
        print("正在准备数据...")
        prepare_stl_data_for_plots()

    # 读取数据
    X, y = read_data(filename)
    X = X.reshape(-1, size, size)  # 修改为单通道

    # 找到飞机和鸟的索引
    idx_plane_all = np.where(y == -1)[0]  # 飞机的所有索引
    idx_bird_all = np.where(y == 1)[0]  # 鸟的所有索引

    # 选择前两个索引
    idx_plane_selected = idx_plane_all[:2]
    idx_bird_selected = idx_bird_all[:2]

    # 创建保存目录
    os.makedirs('figures/single', exist_ok=True)

    # 保存每张图像
    for i, idx in enumerate(idx_plane_selected):
        plt.figure(figsize=(2, 2))
        img = X[idx]
        img = (img - img.min()) / (img.max() - img.min())  # 归一化
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig(f'figures/single/stl-96x96_plane_{i + 1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    for i, idx in enumerate(idx_bird_selected):
        plt.figure(figsize=(2, 2))
        img = X[idx]
        img = (img - img.min()) / (img.max() - img.min())  # 归一化
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig(f'figures/single/stl-96x96_bird_{i + 1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"已保存4张STL图像到 figures/single/")


def prepare_skin_cancer_data_for_plots():
    # 获取项目根目录的绝对路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # 构建输入和输出文件夹的绝对路径
    input_folder = os.path.join(project_root, "datasets", "skin_cancer_kh")
    output_folder = os.path.join(project_root, "paper", "plots", "datasets-for-plots", "skin_cancer_kh")

    # 确保输入文件夹存在
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"错误: 输入文件夹 '{input_folder}' 不存在")

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 只处理224x224的数据
    filename = "skin_cancer_kh_224x224_train.csv"
    if filename.endswith("_train.csv"):
        input_path = os.path.join(input_folder, filename)
        output_filename = filename.replace("_train.csv", "_train-20.csv")
        output_path = os.path.join(output_folder, output_filename)
        try:
            with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
                # 直接复制前 20 行
                for _ in range(20):
                    line = infile.readline()
                    if line:
                        outfile.write(line)
                    else:
                        break  # 如果文件行数少于 20，则提前结束

            print(f"已创建文件：{output_path}")
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")

    print("数据准备完成")


def plot_skin_cancer():
    # 获取项目根目录的绝对路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    size = 224  # 只处理224x224的图像
    data_folder = os.path.join(project_root, "paper", "plots", "datasets-for-plots", "skin_cancer_kh")

    # 检查是否存在用于绘图的数据集
    filename = os.path.join(data_folder, f"skin_cancer_kh_{size}x{size}_train-20.csv")
    if not os.path.exists(filename):
        print(f"未找到用于绘图的 Skin Cancer 数据集: {filename}")
        print("正在准备数据...")
        prepare_skin_cancer_data_for_plots()

    # 读取数据
    X, y = read_data(filename)
    X = X.reshape(-1, size, size)  # 修改为单通道

    # 找到良性和恶性肿瘤的索引
    idx_benign_all = np.where(y == -1)[0]  # 良性肿瘤的所有索引
    idx_malignant_all = np.where(y == 1)[0]  # 恶性肿瘤的所有索引

    # 选择前两个索引
    idx_benign_selected = idx_benign_all[:2]
    idx_malignant_selected = idx_malignant_all[:2]

    # 创建保存目录
    os.makedirs('figures/single', exist_ok=True)

    # 保存每张图像
    for i, idx in enumerate(idx_benign_selected):
        plt.figure(figsize=(2, 2))
        img = X[idx]
        img = (img - img.min()) / (img.max() - img.min())  # 归一化
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig(f'figures/single/skin-cancer-224x224_benign_{i + 1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    for i, idx in enumerate(idx_malignant_selected):
        plt.figure(figsize=(2, 2))
        img = X[idx]
        img = (img - img.min()) / (img.max() - img.min())  # 归一化
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig(f'figures/single/skin-cancer-224x224_malignant_{i + 1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"已保存4张Skin Cancer图像到 figures/single/")


def prepare_dtd_data_for_plots():
    # 获取项目根目录的绝对路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # 构建输入和输出文件夹的绝对路径
    input_folder = os.path.join(project_root, "datasets", "dtd_kh")
    output_folder = os.path.join(project_root, "paper", "plots", "datasets-for-plots", "dtd_kh")

    # 确保输入文件夹存在
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"错误: 输入文件夹 '{input_folder}' 不存在")

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 只处理224x224的数据
    filename = "dtd_kh_224x224_train.csv"
    if filename.endswith("_train.csv"):
        input_path = os.path.join(input_folder, filename)
        output_filename = filename.replace("_train.csv", "_train-20.csv")
        output_path = os.path.join(output_folder, output_filename)
        try:
            with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
                # 直接复制前 20 行
                for _ in range(20):
                    line = infile.readline()
                    if line:
                        outfile.write(line)
                    else:
                        break  # 如果文件行数少于 20，则提前结束

            print(f"已创建文件：{output_path}")
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")

    print("数据准备完成")


def plot_dtd():
    # 获取项目根目录的绝对路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    size = 224  # 只处理224x224的图像
    data_folder = os.path.join(project_root, "paper", "plots", "datasets-for-plots", "dtd_kh")

    # 检查是否存在用于绘图的数据集
    filename = os.path.join(data_folder, f"dtd_kh_{size}x{size}_train-20.csv")
    if not os.path.exists(filename):
        print(f"未找到用于绘图的 DTD 数据集: {filename}")
        print("正在准备数据...")
        prepare_dtd_data_for_plots()

    # 读取数据
    X, y = read_data(filename)
    X = X.reshape(-1, size, size)  # 修改为单通道

    # 找到方格纹理和点状纹理的索引
    idx_chequered_all = np.where(y == -1)[0]  # 方格纹理的所有索引
    idx_dotted_all = np.where(y == 1)[0]  # 点状纹理的所有索引

    # 选择前两个索引
    idx_chequered_selected = idx_chequered_all[:2]
    idx_dotted_selected = idx_dotted_all[:2]

    # 创建保存目录
    os.makedirs('figures/single', exist_ok=True)

    # 保存每张图像
    for i, idx in enumerate(idx_chequered_selected):
        plt.figure(figsize=(2, 2))
        img = X[idx]
        img = (img - img.min()) / (img.max() - img.min())  # 归一化
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig(f'figures/single/dtd-224x224_chequered_{i + 1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    for i, idx in enumerate(idx_dotted_selected):
        plt.figure(figsize=(2, 2))
        img = X[idx]
        img = (img - img.min()) / (img.max() - img.min())  # 归一化
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig(f'figures/single/dtd-224x224_dotted_{i + 1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"已保存4张DTD图像到 figures/single/")


def main():
    parser = argparse.ArgumentParser(description="绘制不同数据集的图像")
    parser.add_argument('--mnist-cg-8', action='store_true', help='绘制 MNIST-CG-8x8 数据集')
    parser.add_argument('--mnist-cg-32', action='store_true', help='绘制 MNIST-CG-32x32 数据集')
    parser.add_argument('--stl', action='store_true', help='绘制 STL-10 数据集')
    parser.add_argument('--skin_cancer', action='store_true', help='绘制 Skin Cancer 数据集')
    parser.add_argument('--dtd', action='store_true', help='绘制 DTD 数据集')
    parser.add_argument('--all', action='store_true', help='绘制所有数据集')

    args = parser.parse_args()

    if args.all:
        plot_mnist_cg(8)
        plot_mnist_cg(32)
        plot_stl()
        plot_skin_cancer()
        plot_dtd()
    else:
        if args.mnist_cg_8:
            plot_mnist_cg(8)
        if args.mnist_cg_32:
            plot_mnist_cg(32)
        if args.stl:
            plot_stl()
        if args.skin_cancer:
            plot_skin_cancer()
        if args.dtd:
            plot_dtd()


if __name__ == "__main__":
    main()
