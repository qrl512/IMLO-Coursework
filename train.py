#this must: load the train dataset, create the model, train for 30 epochs, save model to .pth file
import torch.nn as nn
import torch
from data import get_datasets
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
    optimiser = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0) #current values just the same as the Pytorch class definition, need to investigate whether i want momentum or not

    best_validation_accuracy = 0 #update this to keep track of the best validation accuracy throughout all epochs


    epochs = 30 #coursework doc specifies 30, make sure it stays like this
    for epoch in range(epochs):
        #loop through each epoch
        #training phase
        model.train()
        train_loss = 0 #accumulate over epochs
        train_correct = 0 

        for labels, images in train_loader:
            #forward pass first
            outputs = model(images)
            loss = loss_function(outputs, labels) #from lecture slides

            #backward passs - model needs to learn and weights must be updated, use week 9 backrop info

            #get rid of the memory of the precvious mistakes, gradients
            optimiser.zero_grad()
            #then calculate how much each weight contributed to the wrong answer, measure how wrong they were
            loss.backward()
            #change the weights based on how wrong they were before, gradient descent update in week 8 
            optimiser.step() #adjust the weights based on what has been learnt

            #track how well the model is doing during training, add each batch's error to the total error to keep track
            train_loss += loss.item() #this batch's loss
            #compare the predictions to the actual labels and count the number of correct ones
            highest_values, predicted = torch.max(outputs, 1)
            #need to get predicted values to be able to then comapre with labels
            train_correct += (predicted == labels).sum().item() #sum of the times where predicted value = actual label, will probs need .item() to convert to python number

        #need to calculate training accuracy - number of images model predicted correctly divided by number of images in training dataset as a percentage
        training_accuracy = 100 * train_correct / len(train_loader.dataset)
        average_train_loss = train_loss / len(train_loader)

        #validation phase
        validation_correct = 0 #number of images the model predicts correctly

        for images, labels in val_loader:
            outputs = model(images)
            highest_values, predicted = torch.max(outputs, 1)
            validation_correct += (predicted == labels).sum().item()

        #calculate validation accuracy - number of images model predicted correctly divided by total number of images in validation set as a percentage
        val_accuracy = 100 * validation_correct / len(val_loader.dataset) #want to aim for 70-90%

        #print the training loss, accuracy and validation accuracy for each epoch as per coursework request
        printf(f"Epoch {epoch + 1}/{epochs}" - Training loss: {average_train_loss}, Training accuracy: {training_accuracy}%, Validation accuracy: {val_accuracy}%)


        #save the best model to a bestmodel.pth (not made yet)

        #save the final model to model.pth

