import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from .wonder import Wonder


class RealAmplitudesCircuit(nn.Module):
    """使用 PennyLane 实现的 Real Amplitudes Circuit 量子层"""

    def __init__(self, n_qubits, spectrum_layers, use_noise):
        super().__init__()
        self.n_qubits = n_qubits
        self.spectrum_layers = spectrum_layers
        self.use_noise = use_noise

        # 定义量子设备
        self.dev = qml.device('default.qubit', wires=self.n_qubits)

        # 定义量子电路
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights1, weights2):
            # 对所有量子比特应用 Hadamard 门
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # 第一组 RY 旋转和 CNOT 门
            for k in range(self.spectrum_layers):
                # RY 旋转
                for i in range(self.n_qubits):
                    qml.RY(weights1[k, i], wires=i)

                for i in range(self.n_qubits - 1):
                    for j in range(i + 1, self.n_qubits):
                        qml.CNOT(wires=[i, j])

            # 编码输入数据
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)

            # 第二组 RY 旋转和 CNOT 门
            for i in range(self.n_qubits):
                qml.RY(weights2[i], wires=i)

            for i in range(self.n_qubits - 1):
                for j in range(i + 1, self.n_qubits):
                    qml.CNOT(wires=[i, j])

            # 添加噪声（如果启用）
            if self.use_noise > 0:
                for i in range(self.n_qubits):
                    noise_angle = np.pi * self.use_noise * np.random.rand()
                    qml.RX(noise_angle, wires=i)

            # 返回每个量子比特的 PauliZ 期望值
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # 定义权重形状
        weight_shapes = {
            "weights1": (self.spectrum_layers, self.n_qubits),
            "weights2": (self.n_qubits,)
        }

        # 创建量子层
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.quantum_layer(x)


class QcnnRealAmplitudesDDD(Wonder):
    """基于 Real Amplitudes Circuit 的 Wonder 变体
    
    特点:
    1. 使用 Real Amplitudes Circuit 作为量子处理单元
    2. 结合 CNN 和量子电路的混合架构
    3. 适用于图像分类任务
    """
    
    def __init__(self, hidden_features=4, hidden_layers=2, spectrum_layers=2, use_noise=0,
                 learning_rate=0.0002, weight_decay=1e-3, max_steps=10000, batch_size=32,
                 random_state=42, threshold=0.0, scaling=1.0, plot_identifier=""):
        super().__init__(hidden_features=hidden_features, hidden_layers=hidden_layers,
                        spectrum_layers=spectrum_layers, use_noise=use_noise,
                        learning_rate=learning_rate, weight_decay=weight_decay,
                        max_steps=max_steps, batch_size=batch_size,
                        random_state=random_state, threshold=threshold, scaling=scaling,
                        plot_identifier=plot_identifier)
        
        self.kernel_size = 3

    def _create_model(self):
        """基于原始 QCNN-RealAmplitudesCircuit 的模型实现"""

        class Net(nn.Module):
            def __init__(self, n_qubits, spectrum_layers, use_noise):
                super(Net, self).__init__()
                # 卷积层 - 输入通道为1（灰度图像）
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

                self.fc1 = None

                # 量子层
                self.qc = RealAmplitudesCircuit(
                    n_qubits=n_qubits,
                    spectrum_layers=spectrum_layers,
                    use_noise=use_noise
                )

                self.fc2 = nn.Linear(n_qubits, 1)

                self.n_qubits = n_qubits
                self.flatten_size = None

            def _initialize_fc1(self, flatten_size):
                """动态初始化fc1层"""
                self.flatten_size = flatten_size
                self.fc1 = nn.Linear(flatten_size, self.n_qubits)
                # 将fc1移动到与其他层相同的设备上
                if next(self.parameters()).is_cuda:
                    self.fc1 = self.fc1.cuda()

            def forward(self, x):
                # 卷积层处理
                x = F.relu(F.max_pool2d(self.conv1(x), 2))
                x = F.relu(F.max_pool2d(self.conv2(x), 2))
                x = F.relu(F.max_pool2d(self.conv3(x), 2))
                # 展平
                batch_size = x.shape[0]
                flatten_size = x.shape[1] * x.shape[2] * x.shape[3]
                x = x.view(batch_size, -1)

                # 如果fc1还未初始化，则初始化它
                if self.fc1 is None:
                    self._initialize_fc1(flatten_size)

                # 量子层输入准备
                x = self.fc1(x)
                x = np.pi * torch.tanh(x)  # 将输入缩放到 [-π,π] 范围

                # 量子处理
                quantum_outputs = []
                for i in range(batch_size):
                    quantum_outputs.append(self.qc(x[i]))
                x = torch.stack(quantum_outputs)

                x = F.relu(x)
                x = self.fc2(x.float())
                return x

        self.model_ = Net(
            n_qubits=self.hidden_features,
            spectrum_layers=self.spectrum_layers,
            use_noise=self.use_noise
        )


class QcnnRealAmplitudes(Wonder):
    """
    艰难地实现了 CV 可视化，必须严格遵守以下步骤（原因在于直接使用该类训练时，模型总是不收敛）：
    先用 QcnnRealAmplitudes 训练，得到 joblib 文件
    然后将 QcnnRealAmplitudes 随便改为其他名字，并将该类名从 QcnnRealAmplitudesVisualization 改为 QcnnRealAmplitudes
    再去调用 visualization_CV.py
    """

    def __init__(self, hidden_features=4, hidden_layers=2, spectrum_layers=2, use_noise=0,
                 learning_rate=0.0002, weight_decay=1e-3, max_steps=10000, batch_size=32,
                 random_state=42, threshold=0.0, scaling=1.0, plot_identifier=""):
        super().__init__(hidden_features=hidden_features, hidden_layers=hidden_layers,
                         spectrum_layers=spectrum_layers, use_noise=use_noise,
                         learning_rate=learning_rate, weight_decay=weight_decay,
                         max_steps=max_steps, batch_size=batch_size,
                         random_state=random_state, threshold=threshold, scaling=scaling,
                         plot_identifier=plot_identifier)

        self.kernel_size = 3

    def _create_model(self):
        """基于原始 QCNN-RealAmplitudesCircuit 的模型实现"""

        class Net(nn.Module):
            def __init__(self, n_qubits, spectrum_layers, use_noise, height):
                super(Net, self).__init__()
                self.n_qubits = n_qubits

                # 卷积层 - 输入通道为1（灰度图像）
                self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

                # 直接创建fc1层
                feature_map_size = height // 8
                flatten_size = 64 * feature_map_size * feature_map_size
                self.fc1 = nn.Linear(flatten_size, self.n_qubits)
                if next(self.parameters()).is_cuda:
                    self.fc1 = self.fc1.cuda()

                # 量子层
                self.qc = RealAmplitudesCircuit(
                    n_qubits=self.n_qubits,
                    spectrum_layers=spectrum_layers,
                    use_noise=use_noise
                )

                self.fc2 = nn.Linear(self.n_qubits, 1)

            def extract_features(self, x):
                # 卷积层处理
                x = F.relu(F.max_pool2d(self.conv1(x), 2))
                x = F.relu(F.max_pool2d(self.conv2(x), 2))
                x = F.relu(F.max_pool2d(self.conv3(x), 2))

                # 展平
                batch_size = x.shape[0]
                x = x.view(batch_size, -1)

                # 量子层输入准备
                x = self.fc1(x)
                x = np.pi * torch.tanh(x)  # 将输入缩放到 [-π,π] 范围

                # 量子处理
                quantum_outputs = []
                for i in range(batch_size):
                    quantum_outputs.append(self.qc(x[i]))
                x = torch.stack(quantum_outputs)

                return x

            def forward(self, x):
                if isinstance(x, tuple) and len(x) > 1:
                    data, mode = x
                    features = self.extract_features(data)
                    if mode == 'hidden_states':
                        return features
                else:
                    features = self.extract_features(x)

                x = F.relu(features)
                x = self.fc2(x.float())
                return x

        self.model_ = Net(
            n_qubits=self.hidden_features,
            spectrum_layers=self.spectrum_layers,
            use_noise=self.use_noise,
            height=self.height
        )