#this must: load the train dataset, create the model, train for 30 epochs, save model to .pth file
import torch.nn as nn
from torch.utils.data import DataLoader
from model import PetClassifier

def train():
    #load the datatsets, look to see if i can make this better in the future inside data.py, low priority task though
    train_data, val_data, test_data = get_datasets()

    train_loader = DataLoader(train_data, batch_size = 32, shuffle = True)
    val_loader= DataLoader(val_data, batch_size = 32, shuffle = False)

    #initialise the model, loss and optimiser
    model = PetClassifier()
    #loss? from letures probs NLL loss but need it for multiple classes, add softmax bc my outputs aren't probabilities which NLL needs not raw
    loss_function = nn.NLLLoss() #add log softmax in model -> https://docs.pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html

    #optimiser? lectures showen stochastic gradient descent but is that good enough? torch has SGD which is pretty helpful for me
    optimiser = torch.optim.SGD(model.Parameters(), lr = 0.001, momentum = 0) #current values just the same as the Pytorch class definition, need to investigate whether i want momentum or not



    epochs = 30 #coursework doc specifies 30, make sure it stays like this
    for epoch in range(epochs):
        #loop through each epoch
        #training phase
        model.train()
        train_loss = 0 #accumulate over epochs
        train_correct = 0 

        

        #validation phase

        #save the best model to a bestmodel.pth (not made yet)

        #save the final model to model.pth

