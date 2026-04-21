#this must: load the train dataset, create the model, train for 30 epochs, save model to .pth file
import torch.nn as nn
from torch.utils.data import DataLoader

def train():
    #load the datatsets
    train_data, val_data, test_data = get_datasets()

    train_loader = DataLoader(train_data, batch_size = 32, shuffle = True)
    val_loader= DataLoader(val_data, batch_size = 32, shuffle = False)

    #initialise the model, loss and optimiser
    model = PetClassifier()
    #loss?
    #optimiser?

    epochs = 30 #coursework doc specifies 30, make sure it stays like this
    for epoch in range(epochs):
        #loop through each epoch
        #training phase


        #validation phase

        #save the best model to a bestmodel.pth (not made yet)

        #save the final model to model.pth

