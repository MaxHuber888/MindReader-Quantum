import torch
import pennylane as qml
from model import build_hybrid_model
from data_loaders import load_single_image, load_dataset, get_class_names
from helpers import imshow


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
    input = torch.stack([X,X,X,X])

    # Generate Model Prediction
    model.eval()

    with torch.no_grad():
            input = input.to(device)
            output = model(input)
            _, preds = torch.max(output, 1)

    # Load Dataset for class_names
    # TODO: Clean this up
    class_names = get_class_names(load_dataset())
    print("OUTPUT VEC:",output[0])
    print("HOT INDEX:",preds[0])
    print(class_names)

    # Return Model Prediction
    prediction = class_names[preds[0]]

    return prediction