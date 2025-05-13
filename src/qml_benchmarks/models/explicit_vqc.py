# Refer to: https://github.com/sjerbi/QML-beyond-kernel

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
from .wonder import Wonder


class ExplicitCircuit(nn.Module):
    """PennyLane implementation of Havlivcek's Explicit quantum circuit architecture"""

    def __init__(self, n_qubits, n_layers, heisenberg, use_noise):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.heisenberg = heisenberg
        self.use_noise = use_noise

        # Define quantum device
        self.dev = qml.device('default.qubit', wires=self.n_qubits)

        # Define quantum circuit
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            # IQP-type encoding layer (Havlivcek's encoding)
            for k in range(1):  # One round of encoding
                # Apply Hadamard gates
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)

                # Apply PhaseShift for individual inputs
                for i in range(self.n_qubits):
                    qml.PhaseShift(inputs[k * (self.n_qubits + self.n_qubits * (self.n_qubits - 1) // 2) + i], wires=i)

                # Apply IsingZZ for input products
                count = 0
                base_idx = k * (self.n_qubits + self.n_qubits * (self.n_qubits - 1) // 2)
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qml.IsingZZ(inputs[base_idx + self.n_qubits + count], wires=[i, j])
                        count += 1

            # Variational layers
            param_idx = 0
            if self.heisenberg:
                # Heisenberg model evolution
                for l in range(self.n_layers):
                    for i in range(self.n_qubits):
                        # Apply Heisenberg interactions (XX + YY + ZZ)
                        next_qubit = (i + 1) % self.n_qubits
                        qml.IsingXX(weights[param_idx], wires=[i, next_qubit])
                        qml.IsingYY(weights[param_idx + 1], wires=[i, next_qubit])
                        qml.IsingZZ(weights[param_idx + 2], wires=[i, next_qubit])
                        param_idx += 3
            else:
                # Hardware efficient ansatz
                for l in range(self.n_layers - 1):
                    # Single-qubit rotations
                    for i in range(self.n_qubits):
                        qml.RX(weights[param_idx], wires=i)
                        qml.RY(weights[param_idx + 1], wires=i)
                        qml.RZ(weights[param_idx + 2], wires=i)
                        param_idx += 3

                    # Entangling layer with circular boundary condition
                    for i in range(self.n_qubits - 1):
                        qml.CZ(wires=[i, i + 1])
                    if self.n_qubits > 2:
                        qml.CZ(wires=[0, self.n_qubits - 1])

                # Final rotation layer
                for i in range(self.n_qubits):
                    qml.RX(weights[param_idx], wires=i)
                    qml.RY(weights[param_idx + 1], wires=i)
                    qml.RZ(weights[param_idx + 2], wires=i)
                    param_idx += 3

            # Add noise if enabled
            if self.use_noise > 0:
                for i in range(self.n_qubits):
                    noise_angle = np.pi * self.use_noise * torch.rand(1).item()
                    qml.RX(noise_angle, wires=i)

            # Return expectation value of Z measurement on first qubit
            return [qml.expval(qml.PauliZ(0))]

        # Calculate total number of parameters
        if heisenberg:
            n_params = 3 * n_layers * n_qubits  # 3 parameters per qubit per layer for Heisenberg
        else:
            n_params = 3 * n_layers * n_qubits  # 3 parameters per qubit per layer for rotations

        # Define weight shapes
        weight_shapes = {
            "weights": (n_params,)
        }

        # Create quantum layer
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.quantum_layer(x)


class ExplicitVQC(Wonder):
    """Wonder variant based on Havlivcek's Explicit quantum circuit architecture

    Features:
    1. Uses IQP-type encoding (Havlivcek's encoding) for input data
    2. Supports both hardware-efficient and Heisenberg model variational layers
    3. Direct quantum processing without CNN preprocessing
    """

    def __init__(self, hidden_features=3, hidden_layers=2,
                 use_noise=0, learning_rate=1e-3, weight_decay=5e-4,
                 max_steps=10000, batch_size=32, random_state=42, threshold=0.0,
                 scaling=1.0, plot_identifier=""):
        super().__init__(hidden_features=hidden_features, hidden_layers=hidden_layers,
                         spectrum_layers=None, use_noise=use_noise,
                         learning_rate=learning_rate, weight_decay=weight_decay,
                         max_steps=max_steps, batch_size=batch_size,
                         random_state=random_state, threshold=threshold, scaling=scaling,
                         plot_identifier=plot_identifier)

        self.heisenberg = True

    def _create_model(self):
        """Implementation of the original Explicit model architecture"""

        class Net(nn.Module):
            def __init__(self, hidden_features, n_layers, heisenberg, use_noise):
                super(Net, self).__init__()

                self.hidden_features = hidden_features

                # Direct quantum processing as in original
                self.qc = ExplicitCircuit(
                    n_qubits=self.hidden_features,
                    n_layers=n_layers,
                    heisenberg=heisenberg,
                    use_noise=use_noise
                )

                # Trainable rescaling parameter
                self.rescaling = nn.Parameter(torch.ones(1))
                self.pool_size = (self.hidden_features, self.hidden_features)

            def preprocess_input(self, x):
                """Compute feature vectors for Havlivcek's IQP-type encoding with adaptive pooling."""
                n_samples = x.shape[0]

                # Define adaptive pooling to fixed size
                adaptive_pool = nn.AdaptiveAvgPool2d(self.pool_size)

                # Apply adaptive pooling
                x_pooled = adaptive_pool(x)  # Shape: [batch_size, 1, 2, 2]

                # Flatten the pooled features
                x_flat = x_pooled.view(n_samples, -1)  # Shape: [batch_size, 4]

                # Initialize an empty tensor for concatenated features
                preprocessed = torch.empty((n_samples, 0), device=x.device)

                # Add single-qubit features
                for i in range(self.hidden_features):  # hidden_features features
                    preprocessed = torch.cat([preprocessed, x_flat[:, i].unsqueeze(1)], dim=1)

                # Add two-qubit interaction features
                for m in range(self.hidden_features):  # hidden_features features
                    for j in range(m + 1, self.hidden_features):
                        products = x_flat[:, m] * x_flat[:, j]
                        preprocessed = torch.cat([preprocessed, products.unsqueeze(1)], dim=1)

                return preprocessed  # Shape: [batch_size, hidden_features*2]

            def forward(self, x):
                # Preprocess the input
                preprocessed = self.preprocess_input(x)  # Shape: [batch_size, hidden_features*2]

                batch_size = x.shape[0]

                quantum_outputs = []
                for i in range (batch_size):
                    quantum_outputs.append(self.qc(preprocessed[i]))

                outputs=torch.stack(quantum_outputs)

                # Apply trainable rescaling
                outputs = outputs * self.rescaling  # Shape: [batch_size, 1]

                return outputs  # Shape: [batch_size, 1]

        self.model_ = Net(
            hidden_features=self.hidden_features,
            n_layers=self.hidden_layers,
            heisenberg=self.heisenberg,
            use_noise=self.use_noise
        )
