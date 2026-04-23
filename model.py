#CNN architecture code
#Layers, activation functions (probably ReLU), forward pass, loss + backprop?
#Pytorch models need to inherit from nn.Module (the base class)

#imports
import torch.nn as nn
import torch.nn.functional as F

class PetClassifier(nn.Module):
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
        #thrink image whilst keeping information, reduce spatial size
        self.pool = nn.MaxPool2d(2, 2) #try 2x2 window with stride 2, stride 2 will be quicker but less detail

        #FULLY CONNECTED LAYERS (classification)
        #flattened size -> go from 2d features to 1d vector, after pooling we have 16 height + width and then 64 feature maps so 16 * 16 * 64 = 16384
        #2d spatial data is heightxwidth
        self.fc1 = nn.Linear(64 * 16 * 16, 256) #first fully connected big layer output vector = 256, linear classifier used
        self.fc2 = nn.Linear(256, 37) #previous 256 output vector and 37 classes
        self.dropout = nn.Dropout(0.3) #dropout to prevent overfitting, 50% chance of dropping neuron in layer

    #parent class def forward -> this is data flow and forward pass, if you forget it's on the pytorch website
    def forward(self, x):
        #need to define x
        #forward pass, data flow where i will connect layers and possibly pooling depending if the 128x128 image size is the right choice

        #Conv1 + ReLU + pool
        x = self.pool(F.relu(self.conv1(x))) #128 -> 64
        #Conv2 + ReLU + pool
        x = self.pool(F.relu(self.conv2(x))) #64 -> 32
        #Conv3 + ReLU + pool
        x = self.pool(F.relu(self.conv3(x))) #32 -> 16

        #flatten from (batch, 64 channels (feature maps), 16 height, 16 width) to (batch, 64*16*16), .view() reshapes tensor without changing any important data
        x = x.view(-1, 64 * 16 *16)

        #fully connected layers with droupout (no overfitting)
        x = F.relu(self.fc1(x)) #forward through first fully connected layer
        x = self.dropout(x) #apply dropout in forwad pass
        x = self.fc2(x) #classification layer -> final layer
        x = F.log_softmax(x, dim = 1) #raw outputs -> probabiltiies for NLL

        return x #return the output

#LAYERS
# Convolutional layers: extract features from the image like textures, shapes and edges
# Pooling layer: reduce image size and see if it is good generalisation (model predicts correctly for data)
# Flattening layer: convert image's features into 1d vector -> conv layers output 3d features and conneted layres need 1d vectors
# Connected layers (full connected): perform the actual classification
# Final output (not really a layer): output vector (shoulod be 37 numbers as each number corresponds to one animal breed (one class)