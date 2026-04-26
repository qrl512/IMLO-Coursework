# must handle dataset loading of the Oxford Pets dataset
# apply data augmentation to the training set 

import torch #core pytorch
from torchvision import datasets, transforms #transforms is an image processing tool
from torch.utils.data import DataLoader 
from torch.utils.data import random_split
from torch.utils.data import Subset

def get_datasets():
    # .Compose transforms multiple steps into one pipeline so that they run in order

    #training transforms:
    # - resize images to a fixed 128x128 pixel size for consistency
    # - apply data augmentation (flip and rotation) to improve generalisation
    # - convert images to tensors (numeric arrays) and normalise the pixel values
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        #transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(), #convert images into tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    #evaluation transforms:
    # - no augmentation to esnsure consistent and fair evaluation
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    #load the dataset

    #load the full dataset combining training and validation to later split
    full_dataset = datasets.OxfordIIITPet(
        root = "./data",
        split = "trainval",
        transform = train_transform,
        target_types = "category",
        download = True
    )

    #training/validation split:
    # - split into train and validation with 80% training and 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_data, val_data = random_split((full_dataset), [train_size, val_size])
    #change validation transforma after the split
    val_data.dataset.transform = eval_transform
    
    #load the test datatset with evaluation transform so it has no augmentation
    test_dataset = datasets.OxfordIIITPet(
        root = "./data",
        split = "test",
        transform = eval_transform,
        target_types = "category",
        download = True
    )

    return train_data, val_data, test_dataset