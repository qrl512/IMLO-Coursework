# must handle dataset loading of the Oxford Pets dataset

import torch #core pytorch
from torchvision import datasets, transforms #transforms is an image processing tool
from torch.utils.data import DataLoader 
from torch.utils.data import random_split
from torch.utils.data import Subset

def get_datasets():
    #DEBUGGING
    # RAW LABEL CHECK (no transforms)
    raw_check = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    transform=None,
    target_types="category",
    download=True
)

    raw_labels = [raw_check[i][1] for i in range(len(raw_check))]
    print(f"RAW - Min label: {min(raw_labels)}")
    print(f"RAW - Max label: {max(raw_labels)}")
    print(f"RAW - Num classes: {len(set(raw_labels))}")

    # .Compose transforms multiple steps into one pipeline so that they run in order, maybe look more into if this is a good choice
    # Load the raw dataset withouth any transforming first 
    #define the transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomRotation(15),
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
        target_types = "category",
        download = True
    )

    #DEBUGGING
    #LABEL CHECK
    print("Checking labels in full_dataset:")
    sample_labels = []
    for i in range(min(100, len(full_dataset))):
        _, label = full_dataset[i]
        sample_labels.append(label)
    print(f"Min label value: {min(sample_labels)}")
    print(f"Max label value: {max(sample_labels)}")
    print(f"Unique labels (first 20): {sorted(set(sample_labels))[:20]}")
    print(f"Total unique labels in sample: {len(set(sample_labels))}")


    #split into train and validation with 80% training and 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_data, val_data = random_split((full_dataset), [train_size, val_size])
    #cahnge validation transforma after the split
    val_data.dataset.transform = eval_transform
    
    #load teh test datatset with eval_transform
    test_dataset = datasets.OxfordIIITPet(
        root = "./data",
        split = "test",
        transform = eval_transform,
        target_types = "category",
        download = True
    )

    return train_data, val_data, test_dataset