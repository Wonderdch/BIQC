import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from typing import Tuple
from .wonder import Wonder


class QuantumConvLayer(nn.Module):
    """量子卷积层实现 - 使用 quanvolutional5 电路结构"""

    def __init__(self, kernel_size: int, n_qubits: int, spectrum_layers: int, use_noise: float):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_qubits = n_qubits
        self.spectrum_layers = spectrum_layers
        self.use_noise = use_noise

        # 定义量子设备
        self.dev = qml.device('default.qubit', wires=self.n_qubits)

        # 定义量子电路
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            # 处理批量输入
            batch_size = inputs.shape[0]
            
            # 第一个 XZ 层
            for i in range(self.n_qubits):
                qml.RX(weights[i], wires=i)
                qml.RZ(weights[self.n_qubits + i], wires=i)

            # 四个 decrease_r 层
            k = 2 * self.n_qubits  # 权重索引
            for control_idx in range(self.n_qubits - 1, self.n_qubits - 5, -1):
                # 对每个控制比特，与其他所有比特进行受控旋转
                controlled_qubits = list(range(self.n_qubits))
                controlled_qubits.remove(control_idx)
                for target in controlled_qubits:
                    qml.CRZ(weights[k], wires=[control_idx, target])
                    k += 1

            # 编码输入数据 - 批量处理
            for i in range(self.n_qubits):
                qml.RY(inputs[:, i], wires=i)  # 注意这里改为处理所有批次

            # 最后的 XZ 层
            for i in range(self.n_qubits):
                qml.RX(weights[k + i], wires=i)
                qml.RZ(weights[k + self.n_qubits + i], wires=i)

            # 添加噪声（如果启用）
            if self.use_noise > 0:
                for i in range(self.n_qubits):
                    noise_angles = np.pi * self.use_noise * np.random.rand(batch_size)  # 为每个批次生成噪声
                    qml.RX(noise_angles, wires=i)

            # 返回所有批次的测量结果
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # 计算权重数量
        n = self.n_qubits
        total_params = n * n + 3 * n  # 与 quanvolutional5 保持一致

        # 定义权重形状
        weight_shapes = {
            "weights": (total_params,)
        }

        # 创建量子层
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def add_padding(self, x: torch.Tensor, padding: Tuple[int, int]) -> torch.Tensor:
        """添加填充"""
        n, m = x.shape[-2:]
        r, c = padding
        padded = torch.zeros((*x.shape[:-2], n + r * 2, m + c * 2), device=x.device)
        padded[..., r:n + r, c:m + c] = x
        return padded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        
        # 计算需要的填充，确保填充后的尺寸能被kernel_size整除
        target_h = ((orig_h + self.kernel_size - 1) // self.kernel_size) * self.kernel_size
        target_w = ((orig_w + self.kernel_size - 1) // self.kernel_size) * self.kernel_size
        
        pad_h = target_h - orig_h
        pad_w = target_w - orig_w
        
        # 平均分配填充到四周
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # 使用PyTorch的pad函数进行填充
        if pad_h > 0 or pad_w > 0:
            # 注意：PyTorch pad顺序是 (左,右,上,下)
            x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        
        padded_h, padded_w = x.shape[-2], x.shape[-1]
        # print(f"Input: {orig_h}×{orig_w}, Padded: {padded_h}×{padded_w}")
        
        # 输出张量初始化
        out_h = padded_h // self.kernel_size
        out_w = padded_w // self.kernel_size
        out = torch.zeros((batch_size, self.n_qubits, out_h, out_w), device=x.device)
        
        # 批量处理所有子图像
        for i in range(0, padded_h, self.kernel_size):
            for j in range(0, padded_w, self.kernel_size):
                # 安全检查：确保我们不会超出边界
                if i + self.kernel_size > padded_h or j + self.kernel_size > padded_w:
                    continue
                    
                # 提取patch并验证形状
                patch = x[:, 0, i:i + self.kernel_size, j:j + self.kernel_size]
                flat_patch = patch.reshape(batch_size, -1)
                
                expected_features = self.kernel_size * self.kernel_size
                if flat_patch.shape[1] != expected_features:
                    # 不应该发生，但如果发生，填充到正确大小
                    print(f"Warning: Unexpected patch shape {flat_patch.shape} at ({i},{j})")
                    adjusted = torch.zeros((batch_size, expected_features), device=x.device)
                    adjusted[:, :flat_patch.shape[1]] = flat_patch
                    flat_patch = adjusted
                
                # 归一化
                flat_patch = torch.where(
                    torch.all(flat_patch == 0, dim=1, keepdim=True),
                    torch.tensor([[1.] + [0.] * (expected_features - 1)], device=x.device).repeat(batch_size, 1),
                    flat_patch / torch.norm(flat_patch, dim=1, keepdim=True)
                )
                
                # 批量量子处理
                quantum_outputs = self.quantum_layer(flat_patch)
                
                # 存储结果
                out[:, :, i // self.kernel_size, j // self.kernel_size] = torch.as_tensor(
                    quantum_outputs, device=x.device, dtype=x.dtype)
        
        return out


class DynamicQNN(Wonder):
    """动态量子神经网络模型"""
    def __init__(self, kernel_size=2, spectrum_layers=2, use_noise=0,
                 learning_rate=0.001, weight_decay=1e-3, 
                 max_steps=10000, batch_size=32, plot_identifier=""):
        super().__init__(hidden_features=4, hidden_layers=1,
                         spectrum_layers=spectrum_layers, use_noise=use_noise,
                         learning_rate=learning_rate, weight_decay=weight_decay,
                         max_steps=max_steps, batch_size=batch_size,
                         random_state=42, threshold=0.0, scaling=1.0,
                         plot_identifier=plot_identifier)

        self.kernel_size = kernel_size  # 量子卷积核大小
        self.n_qubits = self.kernel_size * self.kernel_size  # 量子比特数量

    def _create_model(self):
        """创建混合量子-经典模型"""

        class Net(nn.Module):
            def __init__(self, kernel_size, n_qubits, spectrum_layers, use_noise):
                super(Net, self).__init__()
                # 量子卷积层
                self.quantum_conv = QuantumConvLayer(
                    kernel_size=kernel_size,
                    n_qubits=n_qubits,
                    spectrum_layers=spectrum_layers,
                    use_noise=use_noise
                )

                self.flatten = nn.Flatten()
                self.fc1 = None  # 动态初始化
                self.fc2 = nn.Linear(1024, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.4)

            def _initialize_fc1(self, flatten_size):
                """动态初始化第一个全连接层"""
                self.fc1 = nn.Linear(flatten_size, 1024)
                if next(self.parameters()).is_cuda:
                    self.fc1 = self.fc1.cuda()

            def forward(self, x):
                # 量子卷积处理
                x = self.quantum_conv(x)

                # 展平
                x = self.flatten(x)

                # 动态初始化fc1（如果需要）
                if self.fc1 is None:
                    self._initialize_fc1(x.shape[1])

                # 经典神经网络处理
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                return x

        self.model_ = Net(
            kernel_size=self.kernel_size,
            n_qubits=self.n_qubits,
            spectrum_layers=self.spectrum_layers,
            use_noise=self.use_noise
        )
