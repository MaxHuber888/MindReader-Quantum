import os
import torch
from torchvision import datasets, transforms
from PIL import Image
import torchvision.transforms.functional as TF


def load_single_image(path="./image.jpeg"):
    # Set up data transforms
    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Normalize input channels using mean values and standard deviations of ImageNet.
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    # Load image
    img = Image.open(path)
    X = data_transforms(img)
    return X


def load_dataset(data_dir="./data"):
    # Set up data transforms
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # Normalize input channels using mean values and standard deviations of ImageNet.
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Load image dataset
    image_datasets = {
        x if x == "train" else "validation": datasets.ImageFolder(
            os.path.join(data_dir, x), data_transforms[x]
        )
        for x in ["train", "val"]
    }

    return image_datasets


def get_dataset_sizes(image_datasets):
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "validation"]}
    return dataset_sizes


def get_class_names(image_datasets):
    class_names = image_datasets["train"].classes
    return class_names


def get_dataloaders(image_datasets, batch_size):
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
        for x in ["train", "validation"]
    }
    return dataloaders
