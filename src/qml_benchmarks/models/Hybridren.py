import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler


class QuantumConvLayer(nn.Module):
    def __init__(self, kernel_size=2, stride=2, n_layers=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_qubits = kernel_size * kernel_size
        self.n_layers = n_layers

        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.rand_params = nn.Parameter(torch.randn(n_layers, self.n_qubits))

        @qml.qnode(self.dev, interface="torch")
        def circuit(phi):
            wires = list(range(self.n_qubits))

            # 编码 4 个经典输入值
            for i in wires:
                qml.RY(torch.pi * phi[i], wires=i)

            # 随机量子电路
            qml.RandomLayers(self.rand_params, wires=wires)

            # 测量产生 4 个经典输出值
            return [qml.expval(qml.PauliZ(j)) for j in wires]

        self.circuit = circuit

    # def forward(self, image):
    #     batch_size, _, height, width = image.shape
    #     out_height = (height - self.kernel_size) // self.stride + 1
    #     out_width = (width - self.kernel_size) // self.stride + 1
    #     out = torch.zeros((batch_size, out_height, out_width, self.n_qubits), dtype=torch.float32)
    #
    #     for b in range(batch_size):
    #         for j in range(0, height - self.kernel_size + 1, self.stride):
    #             for k in range(0, width - self.kernel_size + 1, self.stride):
    #                 q_input = image[b, 0, j:j + self.kernel_size, k:k + self.kernel_size].reshape(-1)
    #                 q_results = self.circuit(q_input)
    #                 out[b, j // self.stride, k // self.stride] = torch.tensor(q_results, dtype=torch.float32)
    #
    #     return out.permute(0, 3, 1, 2)  # 调整通道顺序以匹配 PyTorch 约定

    def forward(self, image):
        batch_size, _, height, width = image.shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1

        # import time

        # 记录开始时间
        # start_time = time.time()

        # 使用 F.unfold 来创建滑动窗口
        unfolded = F.unfold(image, kernel_size=self.kernel_size, stride=self.stride)
        unfolded = unfolded.transpose(1, 2).contiguous().view(-1, self.kernel_size * self.kernel_size)

        # 创建输出张量
        out = torch.zeros((batch_size * out_height * out_width, self.n_qubits), dtype=torch.float32,
                          device=image.device)

        # 使用 for 循环处理每个窗口，因为量子电路可能无法批量处理
        for i, window in enumerate(unfolded):
            q_results = self.circuit(window)
            out[i] = torch.stack(q_results).to(dtype=torch.float32)
            # out[i] = torch.tensor(q_results, dtype=torch.float32, device=image.device)

        # 重塑输出张量
        out = out.view(batch_size, out_height, out_width, self.n_qubits)

        # 记录结束时间并计算耗时
        # end_time = time.time()
        # execution_time = end_time - start_time
        # CPU 环境下 434s
        # GPU 环境下 384s
        # print(f"量子卷积层执行时间：{execution_time:.4f} 秒")

        return out.permute(0, 3, 1, 2)


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
                # StronglyEntanglingLayers 就是 QFF 中的并行版
                # 参考：https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series/#define-the-parallel-quantum-model
                # 对应论文 Fig.3 的 Parameter layer
                qml.StronglyEntanglingLayers(weights1[i], wires=range(self.in_features), imprimitive=qml.ops.CZ)

                # 对应论文 Fig.3 的 Encoding layer
                # 可以发现输入数据都是直接送到这里面的，所以叫做编码层
                for j in range(self.in_features):
                    qml.RZ(inputs[:, j], wires=j)

            qml.StronglyEntanglingLayers(weights2, wires=range(self.in_features), imprimitive=qml.ops.CZ)

            if self.use_noise != 0:
                for i in range(self.in_features):
                    rand_angles = np.pi + self.use_noise * np.random.rand(batch_size)
                    qml.RX(rand_angles, wires=i)

            return [qml.expval(qml.PauliZ(i)) for i in range(self.in_features)]

        dev = qml.device('default.qubit', wires=in_features)
        # weights1 的参数分别代表了： 
        # 1. 有多少层 StronglyEntanglingLayers
        # 2. 每层 StronglyEntanglingLayers中的 子层数
        # 3. 量子比特数
        # 4. 每层中的多个旋转操作，默认是 3，对应 x,y,z
        weight_shape = {"weights1": (self.n_layer, 1, in_features, 3),
                        "weights2": (1, in_features, 3)}
        self.qnode = qml.QNode(_circuit, dev, diff_method="backprop", interface="torch")
        self.qnn = qml.qnn.TorchLayer(self.qnode, weight_shape)

    def forward(self, x):
        # 保存输入 x 的原始形状以便恢复，除了最后一个维度外，将所有维度保留下来，并将最后一维设为 -1，表示自动计算。
        orgin_shape = list(x.shape[0:-1]) + [-1]

        # qnn 要求输入数据是二维的 (batch_size, in_features)，其中特征只有一个维度
        if len(orgin_shape) > 2:
            # 当输入数据的维度超过 2 时，比如图像的 height 和 width，这些维度需要被展平为一个维度，使得输入数据可以被处理为一个二维张量
            x = x.reshape((-1, self.in_features))
        out = self.qnn(x)
        out = out.reshape(orgin_shape)
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


class HybridrenClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_features=4, hidden_layers=1,
                 spectrum_layers=2, use_noise=0, learning_rate=0.001, weight_decay=1e-3,
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
        self.quantum_conv_layers = 1  # 量子卷积层的数量
        self.out_channels = 4

        # GPU 加速
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            print("GPU 加速已启用")

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

        # 添加量子卷积层
        quantum_conv = QuantumConvLayer(kernel_size=2, stride=2, n_layers=self.quantum_conv_layers)
        layers.append(quantum_conv)

        # 添加第一个池化层
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 添加经典卷积层
        # 假设量子卷积层的输出通道数为 self.out_channels
        layers.append(nn.Conv2d(self.out_channels, self.out_channels * 2, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())

        # 添加第二个池化层
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # # 计算量子卷积层输出的特征数
        # conv_out_size = self.height // 2  # 因为 stride=2
        # flattened_size = self.out_channels * conv_out_size * conv_out_size  # 4 是输出通道数

        # 计算展平后的特征数
        with torch.no_grad():
            x = torch.randn(1, 1, self.height, self.width)
            for layer in layers:
                x = layer(x)
            flattened_size = x.numel()

        # 添加一个线性层来调整维度
        layers.append(nn.Flatten())
        layers.append(nn.Linear(flattened_size, self.hidden_features))

        for _ in range(self.hidden_layers):
            layers.append(HybridLayer(self.hidden_features, self.hidden_features, self.spectrum_layers, self.use_noise))

        classifier =  nn.Linear(self.hidden_features, 1)       

        # 创建一个自定义的模型类来替代 nn.Sequential
        class HybridModel(nn.Module):
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

        self.model_ = HybridModel(layers, classifier)
        # return DebugSequential(*layers)

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

        # 将标签转换为0和1
        y = (y + 1) / 2

        # 数据预处理
        X = self.transform(X)
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y).view(-1, 1)

        # 创建和训练模型
        self._create_model()
        if self.use_cuda:
            self.model_ = self.model_.cuda()

        criterion = nn.BCEWithLogitsLoss()

        # optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # 打印模型信息
        self.print_model_info()

        self.loss_history_ = []
        step = 0
        while step < self.max_steps:
            # 使用 get_random_batch 方法获取批次数据
            batch_X, batch_y = self.get_random_batch(X_tensor, y_tensor, self.batch_size)

            if self.use_cuda:
                batch_X = batch_X.cuda()
                batch_y = batch_y.cuda()

            outputs = self.model_(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.loss_history_.append(loss.item())  # 记录损失值

            # if step % 10 == 0:
            #     self.loss_history_.append(loss.item())  # 记录损失值
            step += 1

        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.take(self.classes_, np.argmax(proba, axis=1))

    def predict_proba(self, X):
        X = self.transform(X)
        X_tensor = torch.tensor(X)

        with torch.no_grad():
            if self.use_cuda:
                X_tensor = X_tensor.cuda()

            outputs = torch.sigmoid(self.model_(X_tensor))

        if self.use_cuda:
            outputs = outputs.cpu()
        proba = outputs.numpy()
        return np.column_stack((1 - proba, proba))

    def transform(self, X, preprocess=True):
        # 将 X 从 float64 转换为 float32
        if not np.issubdtype(X.dtype, np.float32):
            X = X.astype(np.float32)

        if preprocess:
            if self.scaler_ is None:
                # 每一列根据该列的极值进行放缩，放缩到 (-π/2, π/2) 范围内
                # 在量子计算中，单量子比特旋转门（如 Rx, Ry, Rz）通常接受 [-π, π] 范围内的角度参数
                # 将数据缩放到 [-π/2, π/2] 可以确保数据在这个有效范围内，同时保留了正负信息
                self.scaler_ = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))

                # 放缩到 (-1, 1) 范围内，其实与 [-π/2, π/2] 的效果基本一致
                # self.scaler_ = MinMaxScaler(feature_range=(-1, 1))

                self.scaler_.fit(X)
            X = self.scaler_.transform(X)
        X = X * self.scaling
        X = np.array(X)

        # 动态计算图像尺寸
        self.height = int(np.sqrt(X.shape[1]))
        self.width = self.height
        assert self.height * self.width == X.shape[1], "输入特征数量必须是完全平方数"

        # 在 MNIST_CG 32x32 图像上测试，二值化效果不佳
        # X = np.heaviside(X - self.threshold, 0.0)  # 二值化输入

        X = X.reshape(-1, 1, self.height, self.width)  # 确保 X 的形状是 (n_samples, height, width)
        return X
