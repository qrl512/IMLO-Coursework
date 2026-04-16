#CNN architecture code
#Layers, activation functions (probably ReLU), forward pass, loss + backprop?
#Pytorch models need to inherit from nn.Module (the base class)

#imports
import torch.nn as nn

class AnimalClassifier(nn.Module):
    #look through the nn.Module parent class to see the constructor
    #must make init call to parent class before assignment of the child class -> pytorch base class info

    def __init__(self):
        #call the parent class constructor
        super().__init__()

        #my network layers

    #parent class def forward -> this is data flow and forward pass
    def forward(self, x):
        #need to define x
        #forward pass, data flow where i will connect layers and possibly pooling depending if the 128x128 image size is the right choice
        
