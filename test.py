# this must: 
# load the test dataset, load the test model and output final test accuracy
import torch
from torch.utils.data import DataLoader
from model import PetClassifier
from data import get_datasets

def test():
    #load test dataset, test daata uses evaluation transforms so there's no augmentation
    train_data, val_data, test_data = get_datasets() #train and validation not used but are returned by the same function

    #create test data loader, shuffle as false to ensure there is no random ordering for testing
    test_loader = DataLoader(test_data, batch_size = 32, shuffle = False)

    #load the trained model, the best model to get best test results
    model = PetClassifier().to(device) #initiliase the model
    model.load_state_dict(torch.load("best_model.pth", map_location = torch.device('cpu'))) #loads the best saved model based on validation accuracy
    model.eval() #set model to evaluation mode to disbale dropout and use running stats for batch normalisation as here we are testing
    
    #test the model

    #counter for correctly classified test samples
    test_correct = 0

    #disable grad tracking so there's no backdrop during testing
    with torch.no_grad():
        for images, labels in test_loader:
            #forward pass -> compute the predictions
            outputs = model(images)
            #get the predicted class
            highest_value, predicted = torch.max(outputs, 1)
            #couint the number of correct predictions
            test_correct += (predicted == labels).sum().item()

    #calculate the test accuracy and print it as %
    test_accuracy = 100 * test_correct / len(test_loader.dataset)
    print(f"Test accurcy: {test_accuracy}%")

    return test_accuracy

if __name__ == "__main__":
    test()