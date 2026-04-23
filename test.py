#this must: load the test dataset, load the test model and output test accuracy, use train.py code as a good starting point
import torch
from torch.utils.data import DataLoader
from model import PetClassifier
from data import get_datasets

def test():
    #load test dataset
    train_data, val_data, test_data = get_datasets()

    #create test data laoder
    test_loader = DataLoader(test_data, batch_size = 32, shuffle = False)

    #load the trained model
    model = PetClassifier()
    model.load_state_dict(torch.load("model.pth", map_location = torch.device('cpu')))
    model.eval()
    
    #test the model
    test_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            highest_value, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()

    #calculate the test accuracy and print it
    test_accuracy = 100 * test_correct / len(test_loader.dataset)
    print(f"Test accurcy: {test_accuracy}%")

    return test_accuracy

if __name__ == "__main__":
    test()