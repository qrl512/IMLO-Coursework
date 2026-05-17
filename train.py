# this must: 
# load the train dataset, create the model, train for 30 epochs, save model to .pth file, save the best model to .pth file
# trains on a CNN on the OxfordIIIT Pet dataset
import torch.nn as nn
import torch
from data import get_datasets
from torch.utils.data import DataLoader
from model import PetClassifier

def train():
    #load the datatsets, the trainval split to train and validation is handled in data.py
    train_data, val_data, test_data = get_datasets()

    #create data loader for train data, shuffle as true to help improve generalisation during training stage
    train_loader = DataLoader(train_data, batch_size = 32, shuffle = True, num_workers = 2, pin_memory = True)
    #DEBUGGING STUFF - debug to verify label are within the expected range, uncomment the next 3 lines to include in code
    #images, labels = next(iter(train_loader))
    #print(labels[:20])
    #print(labels.min(), labels.max())

    #shuffle false so no random ordering for validation
    val_loader= DataLoader(val_data, batch_size = 32, shuffle = False, num_workers = 2, pin_memory = True)

    #initialise the model, loss and optimiser
    model = PetClassifier()

    #loss function:
    # - switched from NLLLoss with LogSoftmax to CrossEntropyLoss as it already combines both and it directly works with raw logtis from the model
    # - common modern practise for CNNs for image classification
    loss_function = nn.CrossEntropyLoss(label_smoothing = 0.1)
    #loss_function = nn.CrossEntropyLoss()

    #optimiser:
    # - switched from SGD (which updated params using fixed LR and momentum) to AdamW
    # - changed to AdamW optimiser as it adjust LR for each parameter which allowed my mdoel to converge faster and more reliably (especially in early training stages)
    # - due to 30 epoch limit and training from scratch, AdamW was better as it didn't need as much fine tuning and reached better performance quicker than SGD
    # - added weight decay (regularisation) to help reduce any overfitting 
    optimiser = torch.optim.AdamW(model.parameters(), lr = 0.0003, weight_decay = 5e-5)

    #scheduler:
    # - keeps the initial learning rate for the first 15 epochs and then reduces it later in training
    # - this allows finer weight updates which helps stabilise training and improve convergence
    def lr_lambda(epoch):
        if epoch < 15:
            return 1.0
        elif epoch < 22:
            return 0.3
        else:
            return 0.09
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    #track the best validation accuracy to be able to save the best performing model of the run
    best_validation_accuracy = 0

    epochs = 30 #coursework doc specifies 30, make sure it stays like this
    for epoch in range(epochs):
        #loop through each epoch
        #training phase
        model.train() #enables dropout and batch normalisation updates
        train_loss = 0 #accumulate over epochs
        train_correct = 0 

        #manual learning rate scheduling:
        # - reduce LR after 15 epochs to 0.0001 from 0.0003 to allow the model to learn fast early adn then refine in later epochs

        #
        #if epoch == 15:
         #   for g in optimiser.param_groups:
          #      g['lr'] = 0.0001

        for images, labels in train_loader:
            #forward pass first
            #make sure images are the correct type for cross entropy loss
            labels = labels.long()

            outputs = model(images)
            #compute loss between predictions and actual true labels
            loss = loss_function(outputs, labels) #from lecture slides

            #backward passs:
            # - model needs to learn and weights must be updated, use week 9 backrop info
            # - clear old gradients, then compute new gradients then update weigths

            #clear old gradients
            optimiser.zero_grad()
            #then calculate how much each weight contributed to the wrong answer, computing the new gradients 
            loss.backward()

            #testing clipping gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            #change the weights based on how wrong they were before, gradient descent update in week 8 
            optimiser.step() #adjust the weights based on what has been learnt

            #track how well the model is doing during training, add each batch's error to the total error to keep track
            train_loss += loss.item() #this batch's loss
            #the predicted class is the index of the highest output logit
            highest_values, predicted = torch.max(outputs, 1)
            #need to get predicted values to be able to then comapre with labels
            train_correct += (predicted == labels).sum().item() #times where predicted value = actual label, will probs need .item() to convert to python number

        scheduler.step() #for cosine/lambda

        #need to calculate training accuracy - number of images model predicted correctly divided by number of images in training dataset as a percentage
        training_accuracy = 100 * train_correct / len(train_loader.dataset)
        average_train_loss = train_loss / len(train_loader)

        #validation phase
        model.eval() #stop batch dropout
        validation_correct = 0 #number of images the model predicts correctly

        #disable gradient tracking, gradients aren't needed during evaluation
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                #the predicted class is the index of the highest output logit
                highest_values, predicted = torch.max(outputs, 1)
                #number of correctly predicted values incrementing validation correct counter
                validation_correct += (predicted == labels).sum().item()

        #calculate validation accuracy - number of images model predicted correctly divided by total number of images in validation set as a percentage
        val_accuracy = 100 * validation_correct / len(val_loader.dataset) #want to aim for 70-90%
        #scheduler.step(val_accuracy) #for reduceLRonplateau

        #print the training loss, accuracy and validation accuracy for each epoch as per coursework request
        print(f"Epoch {epoch + 1}/{epochs}, Training loss: {average_train_loss}, Training accuracy: {training_accuracy}%, Validation accuracy: {val_accuracy}%")


        #save the best model to a bestmodel.pth, best model judged on validation performance
        #if val_accuracy > best_validation_accuracy:
        #    best_validation_accuracy = val_accuracy
        #    torch.save(model.state_dict(), "best_model.pth")

    #save the final model produced in the run to model.pth
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    train()

