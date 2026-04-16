# must handle dataset loading of the Oxford Pets dataset

import torch #core pytorch
from torchvision import datasets, transforms #transforms is an image processing tool
from torch.utils.data import DataLoader 
from torch.utils.data import random_split

def get_datasets():
    # .Compose transforms multiple steps into one pipeline so that they run in order, maybe look more into if this is a good choice
    transform = transforms.Compose([
        transforms.Resize((128, 128)), #resize every image to 128x128 pixels so that all images are the same size, possibly adjust pixel size for resize if 128 isn't right choice
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

    # Slit trainval into training + validation
    train_data, val_data = split_trainval_dataset(train_dataset)

    return train_data, val_data, test_dataset

def split_trainval_dataset(train_dataset):
    #split data 80:20 training:validation for now
    #split randomly for better pattern learning
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size #whatever is left of the dataset

    train_data, val_data = random_split(train_dataset, [train_size, val_size])   
    return train_data, val_data


def test_get_datasets(train_dataset, test_dataset):
    #DataLoader takes dataset and feeds it to a model in batches, batch size is the number of images to load at a time repeatedly until the dataset is finished
    train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True) #shuffle is used to mix images randomly at every epoch so model doesn't learn patterns which aren't real -> good practise
    test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False) #using a small batch size for now, maybe try like 16 or 32 later?? see the effects the batch size has

    #get one batch for train dataset and loader
    images, labels = next(iter(train_loader))
    print("TRAINING DATASET")
    print("batch image shape is: ", images.shape)
    print("batch labels are: ", labels)

    #get one batch for test dataset and loader
    images, labels = next(iter(test_loader))
    print("TESTING DATASET")
    print("batch image shape is: ", images.shape)
    print("batch labels are: ", labels)

if __name__ == "__main__":
    train_dataset, test_dataset = get_datasets()
    test_get_datasets(train_dataset, test_dataset)