# must handle dataset loading of the Oxford Pets dataset

import torch #core pytorch
from torchvision import datasets, transforms #transforms is an image processing tool

def get_datasets():
    # .Compose transforms multiple steps into one pipeline so that they run in order, maybe look more into if this is a good choice
    transform = transforms.Compose([
        transforms.resize((128, 128)), #resize every image to 128x128 pixels so that all images are the same size, possibly adjust pixel size for resize if 128 isn't right choice
        transforms.ToTensor() #convert image into a numeric arrary (tensor) so that the NN can understand it
    ])

    # Load Train Dataset
    train_dataset = datasets.OxfordIIITPet(
        root = "./data", #dataset is in folder 'data' in project, must remember to put datasets in data folder or think how to load them elsewhere
        split = "trainval", #trainval = training + validation
        transform = transform, #applies the processing pipeline from above
        download = True #if no dataset on computer then just download automatically
    )

    # Load Test Dataset
    test_dataset = datasets.OxfordIIITPet(
        root = "./data",
        split = "test",
        transform = transform,
        download = True
    )

    return train_dataset, test_dataset