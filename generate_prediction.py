import torch
import numpy as np
import pennylane as qml
from model import build_hybrid_model
from data_loaders import load_single_image, load_dataset, get_class_names
from helpers import imshow


def softmax(x):
    # Compute softmax values for each set of scores in x.
    e_x = np.exp(x - np.max(x))  # Subtracting max for numerical stability
    return e_x / e_x.sum(axis=0)


def generate_prediction(img_path="./image.jpeg"):
    N_QUBITS = 4
    PARAMETER_FILEPATH = "state.pt"

    # Initialize Pennylane backend
    dev = qml.device("default.qubit", wires=N_QUBITS)

    # Initialize PyTorch config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = build_hybrid_model(
        pennylane_dev=dev,
        device=device,
        n_qubits=N_QUBITS,
        q_depth=6,
        q_delta=0.01
    )

    # Load Model Parameters from Disk
    model.load_state_dict(torch.load(PARAMETER_FILEPATH))

    # Load [image] from Disk
    X = load_single_image(img_path)
    imshow(X)
    input = torch.stack([X, X, X, X])

    # Generate Model Prediction
    model.eval()

    with torch.no_grad():
        input = input.to(device)
        output = model(input)
        _, preds = torch.max(output, 1)

    # Load Dataset for class_names
    class_names = get_class_names(load_dataset())
    print("OUTPUT VEC:", output[0])
    print("HOT INDEX:", preds[0])

    # Return Model Prediction
    probabilities = softmax(output[0].cpu().numpy())
    prediction = {class_name.replace("_", " "): prob for class_name, prob in zip(class_names, probabilities)}

    return prediction
