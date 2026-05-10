#CNN architecture code for image classification on the OxfordIIIT Pet dataset
#Pytorch models need to inherit from nn.Module (the base class)

import torch.nn as nn
import torch
import torch.nn.functional as F

class PetClassifier(nn.Module):
    #look through the nn.Module parent class to see the constructor
    def __init__(self):
        #initialise the parent class constructor (needed for all PyTorch models)
        super().__init__()

        #feature extractor:
        # - series of convoltional blocks
        # - each block increases channel depth and extracts more and more complex features each times 
        # - pattern: conv -> batchnorm -> relu -> conv -> batchnorm -> maxpool
        self.features = nn.Sequential(
            #block 1 -> low level features being extracted (edges, textures)
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            #block 2 -> more complex patterns being extracted
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            #block 3 -> more complex patterns
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            #block 4 -> more complex again
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            #block 5 -> at this point i'm extracting high level more abstract features
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        #global pooling:
        # - this reduces each feature map to a single value
        # - doing this reduces the number of parameters and helps generalisation
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        #classifier:
        # - maps extracted features to class score
        # - there are 37 pet classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            #regularisation to help reduce any overfitting, stuck to 0.3
            nn.Dropout(0.), #optimal coudlve been 0.2 need to check
            nn.Linear(256, 37)
        )

    #parent class def forward -> this is data flow and forward pass, if you forget it's on the pytorch website
    def forward(self, x):
        #forward pass:
        # - extract spatial features using convolutional layers
        x = self.features(x)
        # - reduce spatial dimensions whilst keeping channel information
        x = self.pool(x)
        # -  then classify using fully connected layers
        x = self.classifier(x)

        return x

#LAYERS
# Convolutional layers: extract features from the image like textures, shapes and edges
# Pooling layer: reduce image size and see if it is good generalisation (model predicts correctly for data)
# Flattening layer: convert image's features into 1d vector -> conv layers output 3d features and conneted layres need 1d vectors
# Connected layers (full connected): perform the actual classification
# Final output (not really a layer): output vector (shoulod be 37 numbers as each number corresponds to one animal breed (one class)