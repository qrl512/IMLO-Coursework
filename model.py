#CNN architecture code
#Layers, activation functions (probably ReLU), forward pass, loss + backprop?
#Pytorch models need to inherit from nn.Module (the base class)

#imports
import torch.nn as nn
import torch.nn.functional as F

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

#LAYERS
# Convolutional layers: extract features from the image like textures, shapes and edges
# Pooling layer: reduce image size and see if it is good generalisation (model predicts correctly for data)
# Flattening layer: convert image's features into 1d vector -> conv layers output 3d features and conneted layres need 1d vectors
# Connected layers (full connected): perform the actual classification
# Final output (not really a layer): output vector (shoulod be 37 numbers as each number corresponds to one animal breed (one class)