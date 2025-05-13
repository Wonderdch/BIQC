import torch
import torch.nn as nn
import argparse
import os
from qml_benchmarks.hyperparam_search_utils import read_data
import matplotlib.pyplot as plt
import numpy as np
from qml_benchmarks.models.quanvolutional_neural_network import QuanvolutionalNeuralNetwork


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

    # 遍历输入文件夹中的所有 CSV 文件
    for filename in os.listdir(input_folder):
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

    # STL-10 数据集绘制代码
    resolutions = [96, 64, 32]  # 读取不同分辨率的数据
    X_trains = []
    y_trains = []

    data_folder = os.path.join(project_root, "paper", "plots", "datasets-for-plots", "stl_kh")

    # 检查是否存在用于绘图的 STL-10 数据集
    for res in resolutions:
        filename = os.path.join(data_folder, f"stl_kh_0-1_{res}x{res}_train-20.csv")
        if not os.path.exists(filename):
            print(f"未找到用于绘图的 STL-10 数据集: {filename}")
            print("正在准备数据...")
            prepare_stl_data_for_plots()
            break

    for res in resolutions:
        filename = os.path.join(data_folder, f"stl_kh_0-1_{res}x{res}_train-20.csv")
        X, y = read_data(filename)

        X_trains.append(X.reshape(-1, res, res))  # 修改为单通道
        y_trains.append(y)

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), tight_layout=True)

    # 选择要显示的图像索引
    idx_plane = np.where(y_trains[0] == -1)[0][0]  # 第一个飞机的索引
    idx_bird = np.where(y_trains[0] == 1)[0][0]  # 第一个鸟的索引

    # 绘制每个分辨率的图像
    for i, res in enumerate(resolutions):
        # 飞机图像
        plane_img = X_trains[i][idx_plane]
        plane_img = (plane_img - plane_img.min()) / (plane_img.max() - plane_img.min())
        axes[0][i].imshow(plane_img, cmap='gray')
        axes[0][i].axis('off')
        axes[0][i].set_title(f'{res}x{res} pixels')

        # 鸟图像
        bird_img = X_trains[i][idx_bird]
        bird_img = (bird_img - bird_img.min()) / (bird_img.max() - bird_img.min())
        axes[1][i].imshow(bird_img, cmap='gray')
        axes[1][i].axis('off')
        axes[1][i].set_title(f'{res}x{res} pixels')

    # 保存图像
    plt.savefig(os.path.join(project_root, 'paper', 'plots', 'figures', 'stl_kh.png'))
    plt.close()
    print("STL-10 图像已保存为 stl_kh.png")


def plot_mnist_cg():
    # MNIST_CG 数据集绘制代码
    # note: you need to download the mnist_cg data
    X32, y32 = read_data("datasets-for-plots/mnist_cg/mnist_pixels_3-5_32x32_train-20.csv")
    X32 = np.reshape(X32, (X32.shape[0], 32, 32))

    X16, y16 = read_data("datasets-for-plots/mnist_cg/mnist_pixels_3-5_16x16_train-20.csv")
    X16 = np.reshape(X16, (X16.shape[0], 16, 16))

    X8, y8 = read_data("datasets-for-plots/mnist_cg/mnist_pixels_3-5_8x8_train-20.csv")
    X8 = np.reshape(X8, (X16.shape[0], 8, 8))

    X4, y4 = read_data("datasets-for-plots/mnist_cg/mnist_pixels_3-5_4x4_train-20.csv")
    X4 = np.reshape(X4, (X4.shape[0], 4, 4))

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(17, 9),
                             tight_layout=True)  # Adjust the figsize as needed
    idx3 = 7
    idx5 = -8

    images3 = [-X32[idx3], -X16[idx3], -X8[idx3], -X4[idx3]]
    images5 = [-X32[idx5], -X16[idx5], -X8[idx5], -X4[idx5]]

    # Plot each image in a horizontal line
    for i in range(4):
        axes[0][i].imshow(images3[i], cmap='gray')
        axes[0][i].axis('off')  # Turn off axis labels for clarity
        axes[0][i].set_title(f'{32 // (2 ** i)}x{32 // (2 ** i)} pixels')
        axes[1][i].imshow(images5[i], cmap='gray')
        axes[1][i].axis('off')  # Turn off axis labels for clarity
        axes[1][i].set_title(f'{32 // (2 ** i)}x{32 // (2 ** i)} pixels')

    plt.savefig('figures/mnist_cg.png')
    plt.close()
    print("MNIST_CG 图像已保存为 mnist_cg.png")


def plot_bars_and_stripes():
    # Bars and Stripes 数据集绘制代码
    X, y = read_data('datasets-for-plots/bars_and_stripes/bars_and_stripes_16_x_16_0.5noise.csv')
    fig, axes = plt.subplots(ncols=4, figsize=(8, 8))

    axes[0].axis('off')
    axes[0].imshow(np.reshape(-X[0], (16, 16)), cmap='gray')
    axes[1].axis('off')
    axes[1].imshow(np.reshape(-X[4], (16, 16)), cmap='gray')
    axes[2].axis('off')
    axes[2].imshow(np.reshape(-X[6], (16, 16)), cmap='gray')
    axes[3].axis('off')
    axes[3].imshow(np.reshape(-X[3], (16, 16)), cmap='gray')

    plt.savefig('figures/bars_and_stripes.png', bbox_inches='tight')
    plt.close()
    print("Bars and Stripes 图像已保存为 bars_and_stripes.png")


def plot_quanv_layer():
    # Quanvolutional Layer 绘制代码
    X, y = read_data("datasets-for-plots/mnist_cg/mnist_pixels_3-5_16x16_train-20.csv")

    model = QuanvolutionalNeuralNetwork(n_qchannels=3)
    model.initialize(16 * 16)

    data = np.concatenate((X[-5:], X[:5]))
    X_out = model.batched_quanv_layer(model.transform(data))

    idx3 = 8
    idx5 = 1
    fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(15, 7))
    axes[0][0].axis('off')
    axes[0][0].imshow(np.reshape(-data[idx3], (16, 16)), cmap='gray')
    axes[1][0].axis('off')
    axes[1][0].imshow(np.reshape(-data[idx5], (16, 16)), cmap='gray')
    for i in range(0, 3):
        axes[0][i + 1].imshow(X_out[idx3, :, :, i].T, cmap='gray')
        axes[0][i + 1].axis('off')
        axes[1][i + 1].imshow(X_out[idx5, :, :, i].T, cmap='gray')
        axes[1][i + 1].axis('off')

    plt.savefig("figures/quanv_map.png", bbox_inches='tight')
    plt.close()
    print("Quanvolutional Layer 图像已保存为 quanv_map.png")


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

    # 遍历输入文件夹中的所有 CSV 文件
    for filename in os.listdir(input_folder):
        if filename.endswith("_train.csv"):
            input_path = os.path.join(input_folder, filename)
            output_filename = filename.replace("_train.csv", "_train-20.csv")
            output_path = os.path.join(output_folder, output_filename)
            try:
                with open(input_path, 'r') as infile:
                    # 读取所有行
                    lines = infile.readlines()

                    # 确保文件至少有20行
                    if len(lines) < 20:
                        print(f"警告: 文件 {filename} 行数少于20行，将复制所有行。")
                        selected_lines = lines
                    else:
                        # 选择前10行和后10行
                        selected_lines = lines[:10] + lines[-10:]

                with open(output_path, 'w') as outfile:
                    # 写入选中的行
                    outfile.writelines(selected_lines)

                print(f"已创建文件：{output_path}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

    print("数据准备完成")


def plot_skin_cancer():
    # 获取项目根目录的绝对路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Skin Cancer 数据集绘制代码
    resolutions = [224, 96]  # 读取不同分辨率的数据
    X_trains = []
    y_trains = []

    data_folder = os.path.join(project_root, "paper", "plots", "datasets-for-plots", "skin_cancer_kh")

    # 检查是否存在用于绘图的 Skin Cancer 数据集
    for res in resolutions:
        filename = os.path.join(data_folder, f"skin_cancer_kh_{res}x{res}_train-20.csv")
        if not os.path.exists(filename):
            print(f"未找到用于绘图的 Skin Cancer 数据集: {filename}")
            print("正在准备数据...")
            prepare_skin_cancer_data_for_plots()
            break

    for res in resolutions:
        filename = os.path.join(data_folder, f"skin_cancer_kh_{res}x{res}_train-20.csv")
        X, y = read_data(filename)

        X_trains.append(X.reshape(-1, res, res))  # 修改为单通道
        y_trains.append(y)

    # 创建子图
    fig, axes = plt.subplots(2, len(resolutions), figsize=(17, 9), tight_layout=True)

    # 选择要显示的图像索引
    idx_benign = np.where(y_trains[0] == -1)[0][0]  # 第一个良性肿瘤的索引
    idx_malignant = np.where(y_trains[0] == 1)[0][0]  # 第一个恶性肿瘤的索引

    # 绘制每个分辨率的图像
    for i, res in enumerate(resolutions):
        # 良性肿瘤图像
        benign_img = X_trains[i][idx_benign]
        benign_img = (benign_img - benign_img.min()) / (benign_img.max() - benign_img.min())
        axes[0][i].imshow(benign_img, cmap='gray')
        axes[0][i].axis('off')
        axes[0][i].set_title(f'{res}x{res} pixels (Benign)')

        # 恶性肿瘤图像
        malignant_img = X_trains[i][idx_malignant]
        malignant_img = (malignant_img - malignant_img.min()) / (malignant_img.max() - malignant_img.min())
        axes[1][i].imshow(malignant_img, cmap='gray')
        axes[1][i].axis('off')
        axes[1][i].set_title(f'{res}x{res} pixels (Malignant)')

    # 为整个图添加总标题
    fig.suptitle('Skin Cancer Images: Benign (top) vs Malignant (bottom)', fontsize=16)

    # 保存图像
    plt.savefig(os.path.join(project_root, 'paper', 'plots', 'figures', 'skin_cancer.png'))
    plt.close()
    print("Skin Cancer 图像已保存为 skin_cancer.png")


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

    # 遍历输入文件夹中的所有 CSV 文件
    for filename in os.listdir(input_folder):
        if filename.endswith("_train.csv"):
            input_path = os.path.join(input_folder, filename)
            output_filename = filename.replace("_train.csv", "_train-20.csv")
            output_path = os.path.join(output_folder, output_filename)
            try:
                with open(input_path, 'r') as infile:
                    # 读取所有行
                    lines = infile.readlines()

                    # 确保文件至少有20行
                    if len(lines) < 20:
                        print(f"警告: 文件 {filename} 行数少于20行，将复制所有行。")
                        selected_lines = lines
                    else:
                        # 选择前10行和后10行
                        selected_lines = lines[:10] + lines[-10:]

                with open(output_path, 'w') as outfile:
                    # 写入选中的行
                    outfile.writelines(selected_lines)

                print(f"已创建文件：{output_path}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

    print("DTD 数据准备完成")


def plot_dtd():
    # 获取项目根目录的绝对路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # 创建 dtd 文件夹
    output_folder = os.path.join(project_root, 'paper', 'plots', 'figures', 'dtd')
    os.makedirs(output_folder, exist_ok=True)

    # 定义分辨率
    resolutions = [224, 96]
    X_trains = []
    y_trains = []

    data_folder = os.path.join(project_root, "paper", "plots", "datasets-for-plots", "dtd_kh")

    # 检查是否存在用于绘图的 DTD 数据集
    for res in resolutions:
        filename = os.path.join(data_folder, f"dtd_kh_{res}x{res}_train-20.csv")
        if not os.path.exists(filename):
            print(f"未找到用于绘图的 DTD 数据集: {filename}")
            print("正在准备数据...")
            prepare_dtd_data_for_plots()
            break

    for res in resolutions:
        filename = os.path.join(data_folder, f"dtd_kh_{res}x{res}_train-20.csv")
        X, y = read_data(filename)

        X_trains.append(X.reshape(-1, res, res))  # 修改为单通道
        y_trains.append(y)

    # 计算图像尺寸
    base_size = 3  # 基准尺寸（英寸）
    max_res = max(resolutions)
    fig_sizes = [(base_size * res / max_res) for res in resolutions]
    total_width = max(fig_sizes) * 2 + 1
    total_height = max(fig_sizes) * 2 + 1

    # 获取所有方格纹理和点状纹理的索引
    chequered_indices = np.where(y_trains[0] == -1)[0][:10]  # 获取前10个方格纹理索引
    dotted_indices = np.where(y_trains[0] == 1)[0][:10]  # 获取前10个点状纹理索引

    # 绘制10组图像
    for group_idx in range(10):
        # 创建具有指定尺寸的图
        fig = plt.figure(figsize=(total_width, total_height))
        
        # 创建网格规范
        gs = fig.add_gridspec(2, len(resolutions),
                             width_ratios=[res / max_res for res in resolutions],
                             height_ratios=[1, 1],
                             hspace=0.3, wspace=0.3)

        # 绘制每个分辨率的图像
        for i, res in enumerate(resolutions):
            # 方格纹理图像
            ax1 = fig.add_subplot(gs[0, i])
            chequered_img = X_trains[i][chequered_indices[group_idx]]
            chequered_img = (chequered_img - chequered_img.min()) / (chequered_img.max() - chequered_img.min())
            ax1.imshow(chequered_img, cmap='gray')
            ax1.axis('off')
            ax1.set_title(f'{res}x{res} pixels (Chequered)')

            # 点状纹理图像
            ax2 = fig.add_subplot(gs[1, i])
            dotted_img = X_trains[i][dotted_indices[group_idx]]
            dotted_img = (dotted_img - dotted_img.min()) / (dotted_img.max() - dotted_img.min())
            ax2.imshow(dotted_img, cmap='gray')
            ax2.axis('off')
            ax2.set_title(f'{res}x{res} pixels (Dotted)')

        # 为整个图添加总标题
        fig.suptitle(f'DTD Images Group {group_idx}: Chequered (top) vs Dotted (bottom)', fontsize=16)

        # 保存图像
        output_path = os.path.join(output_folder, f'dtd_group_{group_idx}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"DTD 图像组 {group_idx} 已保存为 dtd_group_{group_idx}.png")

    print(f"所有 DTD 图像已保存到文件夹: {output_folder}")


def plot_dtd_sobel():
    # 获取项目根目录的绝对路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # 定义分辨率
    resolutions = [224, 96]
    X_trains = []
    y_trains = []

    # 创建 Sobel 算子
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    edge_filter_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    edge_filter_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
    edge_filter_x.weight = nn.Parameter(sobel_x, requires_grad=False)
    edge_filter_y.weight = nn.Parameter(sobel_y, requires_grad=False)

    # 创建 MaxPool2d 层
    # max_pool = nn.MaxPool2d(kernel_size=4, stride=4)

    data_folder = os.path.join(project_root, "paper", "plots", "datasets-for-plots", "dtd_kh")

    # 检查是否存在用于绘图的 DTD 数据集
    for res in resolutions:
        filename = os.path.join(data_folder, f"dtd_kh_{res}x{res}_train-20.csv")
        if not os.path.exists(filename):
            print(f"未找到用于绘图的 DTD 数据集: {filename}")
            print("正在准备数据...")
            prepare_dtd_data_for_plots()
            break

    for res in resolutions:
        filename = os.path.join(data_folder, f"dtd_kh_{res}x{res}_train-20.csv")
        X, y = read_data(filename)

        X_trains.append(X.reshape(-1, res, res))  # 修改为单通道
        y_trains.append(y)

    # 创建子图
    fig, axes = plt.subplots(2, len(resolutions), figsize=(17, 9), tight_layout=True)

    # 选择要显示的图像索引
    idx_chequered = np.where(y_trains[0] == -1)[0][0]  # 第一个方格纹理的索引
    idx_dotted = np.where(y_trains[0] == 1)[0][0]  # 第一个点状纹理的索引

    # 绘制每个分辨率的图像
    for i, res in enumerate(resolutions):
        # 方格纹理图像
        img = X_trains[i][idx_chequered]
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        edge_x = edge_filter_x(img_tensor)
        edge_y = edge_filter_y(img_tensor)
        edge_magnitude = torch.sqrt(edge_x.pow(2) + edge_y.pow(2))
        edge_img = edge_magnitude.squeeze().detach().numpy()

        # # 最大池化
        # pooled = max_pool(edge_magnitude)
        # 转换为 numpy 并归一化
        # edge_img = pooled.squeeze().detach().numpy()

        edge_img = (edge_img - edge_img.min()) / (edge_img.max() - edge_img.min())

        axes[0][i].imshow(edge_img, cmap='gray')
        axes[0][i].axis('off')
        axes[0][i].set_title(f'{res}x{res} pixels (Chequered)')

        # 点状纹理图像
        img = X_trains[i][idx_dotted]
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        edge_x = edge_filter_x(img_tensor)
        edge_y = edge_filter_y(img_tensor)
        edge_magnitude = torch.sqrt(edge_x.pow(2) + edge_y.pow(2))
        edge_img = edge_magnitude.squeeze().detach().numpy()

        # # 最大池化
        # pooled = max_pool(edge_magnitude)
        # 转换为 numpy 并归一化
        # edge_img = pooled.squeeze().detach().numpy()

        edge_img = (edge_img - edge_img.min()) / (edge_img.max() - edge_img.min())

        axes[1][i].imshow(edge_img, cmap='gray')
        axes[1][i].axis('off')
        axes[1][i].set_title(f'{res}x{res} pixels (Dotted)')

    # 为整个图添加总标题
    fig.suptitle('DTD Images after Sobel Filter: Chequered (top) vs Dotted (bottom)', fontsize=16)

    # 保存图像
    plt.savefig(os.path.join(project_root, 'paper', 'plots', 'figures', 'dtd_sobel.png'))
    plt.close()
    print("DTD 图像已保存为 dtd_sobel.png")


def main():
    parser = argparse.ArgumentParser(description="绘制不同数据集的图像")
    parser.add_argument('--stl', action='store_true', help='绘制 STL-10 数据集')
    parser.add_argument('--mnist_cg', action='store_true', help='绘制 MNIST_CG 数据集')
    parser.add_argument('--bars_and_stripes', action='store_true', help='绘制 Bars and Stripes 数据集')
    parser.add_argument('--quanv_layer', action='store_true', help='绘制 Quanvolutional Layer')
    parser.add_argument('--skin_cancer', action='store_true', help='绘制 Skin Cancer 数据集')
    parser.add_argument('--dtd', action='store_true', help='绘制 DTD 数据集')
    parser.add_argument('--all', action='store_true', help='绘制所有数据集')
    parser.add_argument('--test', action='store_true', help='绘制测试图像')

    args = parser.parse_args()

    if args.all:
        plot_stl()
        plot_mnist_cg()
        plot_bars_and_stripes()
        plot_quanv_layer()
        plot_skin_cancer()
        plot_dtd()
    else:
        if args.stl:
            plot_stl()
        if args.mnist_cg:
            plot_mnist_cg()
        if args.bars_and_stripes:
            plot_bars_and_stripes()
        if args.quanv_layer:
            plot_quanv_layer()
        if args.skin_cancer:
            plot_skin_cancer()
        if args.dtd:
            plot_dtd()
        if args.test:
            plot_dtd_sobel()


if __name__ == "__main__":
    main()
