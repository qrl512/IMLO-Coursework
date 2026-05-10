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
        transforms.Resize((160, 160)),
        transforms.RandomResizedCrop(160, scale = (0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        #transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(), #convert images into tensorswill
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #evaluation transforms:
    # - no augmentation to esnsure consistent and fair evaluation
    eval_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #create  two separate datasets one for training one for validation
    full_train = datasets.OxfordIIITPet(
        root = "./data",
        split = "trainval",
        transform = train_transform,
        target_types = "category",
        download = True
    )

    full_eval = datasets.OxfordIIITPet(
        root = "./data",
        split = "trainval",
        transform = eval_transform,
        target_types = "category",
        download = True
    )


    #training/validation split:
    # - split into train and validation with 80% training and 20% validation
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size

    #randomly shuffle the dataset indices before splitting to help avoid any bias
    indices = torch.randperm(len(full_train)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data = Subset(full_train, train_indices)
    val_data = Subset(full_eval, val_indices)
    
    #load the test datatset with evaluation transform so it has no augmentation
    test_dataset = datasets.OxfordIIITPet(
        root = "./data",
        split = "test",
        transform = eval_transform,
        target_types = "category",
        download = True
    )

    return train_data, val_data, test_dataset