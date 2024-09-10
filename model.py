import time
import os
import numpy as np

# OpenMP: number of parallel threads.
os.environ["OMP_NUM_THREADS"] = "1"

# PyTorch
import torch
import torch.nn as nn

# Pennylane
import pennylane as qml
import torchvision

torch.manual_seed(42)


def build_hybrid_model(pennylane_dev, device, n_qubits=4, q_depth=6, q_delta=0.01):
    """
    Builds/returns the hybrid model

    Args:
        pennylane_dev (qml.device): The Pennylane Backend
        device (torch.device): PyTorch configuration
        n_qubits (int): Number of qubits
        q_depth (int): Depth of the quantum circuit (number of variational layers)
        q_delta (float): Initial spread of random quantum weights

    Returns:
        torchvision.models.resnet: The hybrid model
    """

    model_hybrid = torchvision.models.resnet18(pretrained=True)

    for param in model_hybrid.parameters():
        param.requires_grad = False

    # Notice that model_hybrid.fc is the last layer of ResNet18
    model_hybrid.fc = DressedQuantumNet(
        n_qubits=n_qubits,
        q_depth=q_depth,
        q_delta=q_delta,
        pennylane_dev=pennylane_dev,
        device=device
    )

    # Use CUDA or CPU according to the "device" object.
    model_hybrid = model_hybrid.to(device)

    return model_hybrid


def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates."""
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis."""
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    '''Layer of CNOTs followed by another shifted layer of CNOT.'''
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


def quantum_net(q_input_features, q_weights_flat, n_qubits, q_depth):
    """
    The variational quantum circuit.
    """

    # Reshape weights
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Start from state |+> , unbiased w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_input_features)

    # Sequence of trainable variational layers
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    # Expectation values in the Z basis
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)


class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self, n_qubits, q_depth, q_delta, pennylane_dev, device):
        """
        Definition of the *dressed* layout.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.q_delta = q_delta
        self.pennylane_dev = pennylane_dev
        self.device = device

        self.pre_net = nn.Linear(512, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))
        self.post_net = nn.Linear(n_qubits, 4)

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        # Create Quantum Net
        qn = qml.QNode(
            func=quantum_net,
            device=self.pennylane_dev,
            interface="torch"
        )

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, self.n_qubits)
        q_out = q_out.to(self.device)
        for elem in q_in:
            q_out_elem = torch.hstack(qn(elem, self.q_params, self.n_qubits, self.q_depth)).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        # return the two-dimensional prediction from the postprocessing layer
        return self.post_net(q_out)
