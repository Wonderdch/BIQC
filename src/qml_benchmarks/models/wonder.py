import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import os
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler


class QuantumLayer(nn.Module):
    def __init__(self, in_features, spectrum_layers, use_noise):
        super().__init__()
        self.in_features = in_features
        self.n_layer = spectrum_layers
        self.use_noise = use_noise

        # 量子层核心代码
        def _circuit(inputs, weights1, weights2):
            batch_size = inputs.shape[0]

            for i in range(self.n_layer):
                qml.StronglyEntanglingLayers(weights1[i], wires=range(self.in_features), imprimitive=qml.ops.CZ)

                for j in range(self.in_features):
                    qml.RZ(inputs[:, j], wires=j)

            qml.StronglyEntanglingLayers(weights2, wires=range(self.in_features), imprimitive=qml.ops.CZ)

            if self.use_noise != 0:
                for i in range(self.in_features):
                    rand_angles = np.pi + self.use_noise * np.random.rand(batch_size)
                    qml.RX(rand_angles, wires=i)

            return [qml.expval(qml.PauliZ(i)) for i in range(self.in_features)]

        dev = qml.device('default.qubit', wires=in_features)

        weight_shape = {"weights1": (self.n_layer, 1, in_features, 3),
                        "weights2": (1, in_features, 3)}
        self.qnode = qml.QNode(_circuit, dev, diff_method="backprop", interface="torch")
        self.qnn = qml.qnn.TorchLayer(self.qnode, weight_shape)

    def forward(self, x):
        origin_shape = list(x.shape[0:-1]) + [-1]

        if len(origin_shape) > 2:
            x = x.reshape((-1, self.in_features))
        out = self.qnn(x)
        out = out.reshape(origin_shape)
        return out


class HybridLayer(nn.Module):
    def __init__(self, in_features, out_features, spectrum_layers, use_noise, bias=True):
        super().__init__()
        self.clayer = nn.Linear(in_features, out_features, bias=bias)
        self.norm = nn.BatchNorm1d(out_features)  # 对每列特征进行 normalization
        self.qlayer = QuantumLayer(in_features=out_features, spectrum_layers=spectrum_layers, use_noise=use_noise)

    def forward(self, x):
        x = self.clayer(x)
        x = self.norm(x)
        out = self.qlayer(x)
        return out


class DebugSequential(nn.Sequential):
    def forward(self, input):
        import time

        for i, module in enumerate(self):
            print(f"Layer {i}: {type(module).__name__}")
            print(f"Module device: {next(module.parameters(), torch.tensor(0)).device}")
            print(f"Input device: {input.device}")
            print(f"Input shape: {input.shape}")

            start_time = time.time()
            input = module(input)
            end_time = time.time()

            layer_time = end_time - start_time
            print(f"Layer {i} 耗时: {layer_time:.4f} 秒")

            print(f"Output device: {input.device}")
            print(f"Output shape: {input.shape}")
            print("-" * 40)
        return input


class Wonder(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_features=4, hidden_layers=1, spectrum_layers=2,
                 use_noise=0, learning_rate=0.001, weight_decay=1e-3,
                 max_steps=10000, batch_size=32, random_state=42,
                 threshold=0.0, scaling=1.0, plot_identifier=""):
        # 存储超参数
        self.hidden_features = hidden_features  # 影响量子比特数
        self.hidden_layers = hidden_layers  # HybridLayer 的层数
        self.spectrum_layers = spectrum_layers  # 每个量子层中 StronglyEntanglingLayers 的层数
        self.use_noise = use_noise
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay  # 添加 L2 正则化
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.threshold = threshold
        self.scaling = scaling
        self.plot_identifier = plot_identifier

        # 确保可重复性
        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)  # 使用 NumPy 的 RandomState 来生成伪随机数
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # 定义数据相关属性
        self.model_ = None
        self.scaler_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.in_features = None
        self.height = None
        self.width = None

        self.loss_history_ = None
        self.n_qubits_ = None

        # 卷积层参数
        self.output_channels = [32, 64]
        # self.output_channels = [16, 32]
        self.kernel_size = 2

        # 收敛检查参数
        self.convergence_interval = 80

    # 用于joblib保存模型时兼容 PyTorch 模型
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unserializable QNode
        if 'model_' in state:
            if hasattr(self.model_, 'state_dict'):
                state['model_state_dict'] = self.model_.state_dict()
            del state['model_']
        return state

    def __setstate__(self, state):
        # First, update the instance's state
        self.__dict__.update(state)
        # Restore the PyTorch model state
        if 'model_state_dict' in state:
            self._create_model()  # Make sure to define the model structure
            self.model_.load_state_dict(state['model_state_dict'])
            del state['model_state_dict']
        self.__dict__.update(state)

    def _create_model(self):
        layers = []

        # 第一个卷积层
        layers.append(nn.Conv2d(1, self.output_channels[0], kernel_size=self.kernel_size, stride=1, padding=1))

        # 第一个池化层
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第二个卷积层
        layers.append(
            nn.Conv2d(self.output_channels[0], self.output_channels[1], kernel_size=self.kernel_size, stride=1,
                      padding=1))

        # 第二个池化层
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 展平层
        layers.append(nn.Flatten())

        # 计算展平后的特征数
        with torch.no_grad():
            x = torch.randn(1, 1, self.height, self.width)
            for layer in layers:
                x = layer(x)
            flattened_size = x.numel()

        # 第一个全连接层
        layers.append(nn.Linear(flattened_size, self.hidden_features))
        layers.append(nn.ReLU())

        # HybridLayer
        for _ in range(self.hidden_layers):
            layers.append(HybridLayer(self.hidden_features, self.hidden_features, self.spectrum_layers, self.use_noise))

        classifier = nn.Linear(self.hidden_features, 1)

        # 创建一个自定义的模型类来替代 nn.Sequential
        class WonderModel(nn.Module):
            def __init__(self, feature_extractor, classifier):
                super().__init__()
                self.feature_extractor = nn.Sequential(*feature_extractor)
                self.classifier = classifier

            def forward(self, x):
                if isinstance(x, tuple) and len(x) > 1:
                    data, mode = x
                    features = self.feature_extractor(data)
                    if mode == 'hidden_states':
                        return features
                else:
                    features = self.feature_extractor(x)

                return self.classifier(features)

        self.model_ = WonderModel(layers, classifier)

    def get_random_batch(self, X, y, batch_size):
        """
        从数据集中随机采样一个批次
        """
        indices = self.rng.randint(0, len(X), size=batch_size)
        return X[indices], y[indices]

    def fit(self, X, y):
        # 初始化数据相关属性
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ == 2, "仅支持二分类问题"

        # 将标签转换为 0 或 1，因为 BCEWithLogitsLoss 期望标签是 0 或 1
        y = (y + 1) / 2

        # 数据预处理
        X = self.transform(X)
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y).view(-1, 1)

        # 创建和训练模型
        self._create_model()
        self.model_.train()

        criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # 打印模型信息
        # self.print_model_info()

        self.loss_history_ = []
        step = 0

        converged = False
        self.converge_step_ = self.max_steps

        while step < self.max_steps:
            # 使用 get_random_batch 方法获取批次数据
            batch_X, batch_y = self.get_random_batch(X_tensor, y_tensor, self.batch_size)

            outputs = self.model_(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            self.loss_history_.append(loss_val)  # 记录损失值
            # if step % 10 == 0:
            #     self.loss_history_.append(loss_val)  # 记录损失值

            # 检查是否有 NaN
            if np.isnan(loss_val):
                print(f"遇到 NaN 损失值，训练中止于第 {step} 步")
                break

            # 收敛检查
            if step > 2 * self.convergence_interval:
                # 计算最近两个区间的统计量
                average1 = np.mean(self.loss_history_[-self.convergence_interval:])
                average2 = np.mean(self.loss_history_[-2 * self.convergence_interval: -self.convergence_interval])
                std1 = np.std(self.loss_history_[-self.convergence_interval:])
                # 判断是否收敛
                if np.abs(average2 - average1) <= std1 / np.sqrt(self.convergence_interval) / 2:
                    self.converge_step_ = step
                    print(f"模型在第 {step} 步收敛")
                    converged = True
                    break

            step += 1

        # 如果达到最大步数仍未收敛，打印信息
        if not converged and step >= self.max_steps:
            print(
                f"收敛失败：Model {self.__class__.__name__} has not converged after the maximum number of {self.max_steps} steps.")

        if self.plot_identifier:
            self.save_loss_figure()
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.take(self.classes_, np.argmax(proba, axis=1))

    def predict_proba(self, X):
        X = self.transform(X)
        X_tensor = torch.tensor(X)

        self.model_.eval()
        with torch.no_grad():
            outputs = torch.sigmoid(self.model_(X_tensor))

        proba = outputs.numpy()
        return np.column_stack((1 - proba, proba))

    def transform(self, X, preprocess=True):
        # 将 X 从 float64 转换为 float32
        if not np.issubdtype(X.dtype, np.float32):
            X = X.astype(np.float32)

        if preprocess:
            if self.scaler_ is None:
                self.scaler_ = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))
                self.scaler_.fit(X)
            X = self.scaler_.transform(X)
        X = X * self.scaling
        X = np.array(X)

        # 动态计算图像尺寸
        self.height = int(np.sqrt(X.shape[1]))
        self.width = self.height
        assert self.height * self.width == X.shape[1], "输入特征数量必须是完全平方数"

        X = X.reshape(-1, 1, self.height, self.width)  # 确保 X 的形状是 (batch_size, 1, height, width)
        return X

    def print_model_info(self):
        if self.model_ is None:
            print("模型尚未初始化。请先调用 fit() 方法。")
            return

        print(self.model_)
        total_params = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        print(f'可训练参数总数：{total_params}')

        # 打印每层的参数数量
        for name, module in self.model_.named_children():
            if isinstance(module, nn.Sequential):
                for i, layer in enumerate(module):
                    params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                    print(f"{name}.{i}: {params} 参数")
            else:
                params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"{name}: {params} 参数")

    def save_loss_figure(self):
        """保存损失曲线到指定路径"""
        # 参数名称映射字典
        param_abbr = {
            'hidden_features': 'HFeature',
            'hidden_layers': 'HLayer',
            'spectrum_layers': 'Spectrum',
            'use_noise': 'Noise',
            'learning_rate': 'LR',
            'weight_decay': 'WD',
            'max_steps': 'Step',
            'batch_size': 'Batch',
            'scaling': 'Scale',
            'random_state': 'Random'
        }

        # 需要排除的参数列表
        exclude_params = {'plot_identifier', 'threshold', 'random_state'}

        plt.figure(figsize=(20, 12))
        plt.plot(self.loss_history_)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.ylim(0, 1.0)

        # 构建文件名
        title = f"{self.__class__.__name__}"

        # 添加标识信息
        title += f"_{self.plot_identifier}"

        # 添加所有超参数信息
        hyperparams = []
        for param_name, param_value in self.get_params().items():
            # 跳过被排除的参数
            if param_name in exclude_params:
                continue

            # 获取参数缩写名
            abbr_name = param_abbr.get(param_name, param_name)  # 如果没有缩写就使用原名

            # 格式化参数值
            if isinstance(param_value, float):
                if param_value < 0.01:
                    hyperparams.append(f"{abbr_name}{param_value:.1e}")
                else:
                    hyperparams.append(f"{abbr_name}{param_value:.2f}")
            else:
                hyperparams.append(f"{abbr_name}{param_value}")

        title += "_" + "_".join(hyperparams)

        plt.title(title)
        plt.tight_layout()

        # 确保保存路径存在
        plt.savefig(os.path.join("./results/loss_figure", f"{title}.png"))
        plt.close()
