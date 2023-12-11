## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)

#         Images Size = 224 x 224
#         O = output
#         W = Input Width
#         S = Stride
#         P = Padding
#         K = Kernel/Filter
        
#         O = (W-K + 2*P)/S + 1
        
        
        self.conv1 = nn.Conv2d(in_channels=1  , out_channels=64 , kernel_size=3, stride=1, padding=0 )
        
#         O = ( (224 - 3 + 0)/1) + 1 = 222 ; O/2 = 222/2  = 111
 
        self.conv2 = nn.Conv2d(in_channels=64 , out_channels=128, kernel_size=3, stride=1, padding=0 )
        
#         O = ( (111 - 3 + 0)/1) + 1 = 109 ; O/2 = 109/2  = 54 (rounded down)
 
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0 )
        
#         O = ( (54 - 3 + 0)/1) + 1 = 52 ; O/2 = 52/2  = 26
 
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0 )
        
#         O = ( (26 - 3 + 0)/1) + 1 = 24 ; O/2 = 24/2  = 12
 
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0 )
        
#         O = ( (12 - 3 + 0)/1) + 1 = 10 ; O/2 = 10/2  = 5
 
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(512  * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024 * 1 * 1, 1024)
        self.fc3 = nn.Linear(1024 * 1 * 1, 136)
        
        self.drop = nn.Dropout(p=0.4)
    
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.drop(self.maxpool(F.relu(self.conv3(x))))
        x = self.maxpool(F.relu(self.conv4(x)))
        x = self.drop(self.maxpool(F.relu(self.conv5(x))))

        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
