import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loaders import load_dataset, get_dataset_sizes, get_dataloaders, get_class_names


# Visualize Model Predictions
def visualize_model(model, device, batch_size, fig_name="Predictions"):
    images_so_far = 0
    _fig = plt.figure(fig_name)
    model.eval()

    # Load data
    data_set = load_dataset()
    dataset_sizes = get_dataset_sizes(data_set)
    class_names = get_class_names(data_set)
    dataloaders = get_dataloaders(data_set, batch_size)

    with torch.no_grad():
        for _i, (inputs, labels) in enumerate(dataloaders["validation"]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(batch_size // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title("[Pred: {}]\n[Label: {}]".format(class_names[preds[j]], class_names[labels[j]]))
                imshow(inputs.cpu().data[j])
                if images_so_far == batch_size:
                    return
        plt.show()


# Show images
def imshow(inp, title=None):
    """Display image from tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Inverse of the initial normalization operation.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
