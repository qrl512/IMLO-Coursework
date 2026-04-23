# must handle dataset loading of the Oxford Pets dataset

import torch #core pytorch
from torchvision import datasets, transforms #transforms is an image processing tool
from torch.utils.data import DataLoader 
from torch.utils.data import random_split

def get_datasets():
    # .Compose transforms multiple steps into one pipeline so that they run in order, maybe look more into if this is a good choice
    # Load the raw dataset withouth any transforming first 

    #define the transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(), #convert image into a numeric arrary (tensor) so that the NN can understand it
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(), #convert image into a numeric arrary (tensor) so that the NN can understand it
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    full_dataset = datasets.OxfordIIITPet(
        root = "./data",
        split = "trainval",
        transform = train_transform,
        download = True
    )

    #split into train and validation with 80% training and 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    #load teh test datatset with eval_transform
    test_dataset = datasets.OxfordIIITPet(
        root = "./data",
        split = "test",
        transform = eval_transform,
        download = True
    )

    return train_data, val_data, test_dataset

def split_trainval_dataset(train_dataset):
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    return train_data, val_data

if __name__ == "__main__":
    train_data, val_data, test_data = get_datasets()