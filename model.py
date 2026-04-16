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
        
        #CONVOLUTIONAL LAYERS
        #input channels = 3 (i have RGB images)
        #ouput channels = 16 each output channel is a feature map, need to decide on a number, 16 seems to be standard chouice
        #kernel size = 3 standard choice for CNN, small enough to capture simple patterns like edges and gradients
        #padding = 1 -> need to stop the image from shrinking after the convolution, this way the size is left unchanged
        #use Pytorch's nn.Conv2d to add 2D convolution over input
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1) #take my rgb image, apply 16 different 3x3 filters and keep the output size the same

        #second convolutional layer:
        #take the 16 feature maps from the first convolutional layer and output 32 feature maps (these are more complex features of the image)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)

        #third convolutional layer (idk if i need a third or if two enough):
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)

        #POOLING LAYER
        #

        #FULLY CONNECTED LAYERS (classification)


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