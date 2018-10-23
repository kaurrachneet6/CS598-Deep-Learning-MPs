# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 21:05:28 2018

@author: Rachneet Kaur
"""
import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#Data Augmentation Step
'''
Transforming the 3*32*32 array to a PIL image format -> Perform Data Transformations 
1. Randomly crops 0.5 to 1 percentage of out of the input 3*32*32 image and resizes to 3*24*24 image 
2. Randomly flip the image horizontally with a probability of 0.5
3. Randomly brighten the image with a factor selected uniformly from [max(0, 1 - brightness), 1 + brightness].
4. Randomly contrast the image with a factor selected uniformly from [max(0, 1 - contrast), 1 + contrast].
-> Transform PIL image back to Tensor format
'''
#Defining the data augmentation transformations
transformations = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      #transforms.ColorJitter(brightness=0.3, contrast=0, saturation=0, hue=0),                              
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train = torchvision.datasets.CIFAR10(root='./CIFARdata', train=True,
                                        download=True, transform=transformations) #data augmentation transformations
data_train = torch.utils.data.DataLoader(train, batch_size=100, #Batch size = 100
                                          shuffle=True, num_workers=0)

#Transforming data and normalizing for the test set
transform_testset = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test = torchvision.datasets.CIFAR10(root='./CIFARdata', train=False,
                                       download=True, transform=transform_testset)
data_test = torch.utils.data.DataLoader(test, batch_size=100,
                                         shuffle=False, num_workers=0)

class CIFAR10Model(nn.Module):
    def __init__(self, num_outputs):
        super(CIFAR10Model, self).__init__() #Specifying the model parameters 
        # input is 3x32x32
        
        #8 convolution layers with ReLU activation 
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride = 1, padding = 2) 
        '''
        No. of output units in a convolution layer = 
        {[No. of input units - Filter Size + 2(Padding)]/Stride} + 1 = 
        (32 - 3 + 2(1))/1 + 1 = 32 
        '''
        
        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride = 1, padding = 2) 
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride = 1, padding = 2) 
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride = 1, padding = 2) 
        self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride = 1, padding = 2) 
        self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 0)     
        self.conv_layer7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 0) 
        self.conv_layer8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 0)  
        '''
        No. of output units in a convolution layer = 
        {[No. of input units - Filter Size + 2(Padding)]/Stride} + 1 = 
        (32 - 3 + 2(1))/1 + 1 = 32 
        '''
        
        #3 Fully connected layers with ReLU activation 
        self.fc1 = nn.Linear(in_features=32*32, out_features= 500) 
        #Image size is 24*24 (due to random resized cropping) with 128 channels from the last convolution layer
        self.fc2 = nn.Linear(in_features=500, out_features= 500)
        self.fc3 = nn.Linear(in_features=500, out_features= num_outputs)      
        
        #Dropout 
        self.dropout = nn.Dropout(p=0.5) # p - Probability of dropping out a neuron
        self.dropout2d = nn.Dropout2d(p=0.5)
        self.batchnorm2d_layer1 = nn.BatchNorm2d(64)
        self.batchnorm2d_layer2 = nn.BatchNorm2d(64) #Batch Normalization
        self.batchnorm2d_layer3 = nn.BatchNorm2d(64)
        self.batchnorm2d_layer4 = nn.BatchNorm2d(64)
        self.batchnorm2d_layer5 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, stride=2) #Max pooling 

    def forward(self, x): #Specifying the NN architecture 
        x = F.relu(self.conv_layer1(x)) #Convolution layers with relu activation
        x = self.batchnorm2d_layer1(x) #Batch normalization 
        x = F.relu(self.conv_layer2(x))
        x = self.pool(x) #Pooling layer
        x = self.dropout2d(x) #Dropout
        x = F.relu(self.conv_layer3(x))
        x = self.batchnorm2d_layer2(x)
        x = F.relu(self.conv_layer4(x))
        x = self.pool(x)
        x = self.dropout2d(x)
        x = F.relu(self.conv_layer5(x))
        x = self.batchnorm2d_layer3(x)
        x = F.relu(self.conv_layer6(x))
        x = self.dropout2d(x)
        x = F.relu(self.conv_layer7(x))
        x = self.batchnorm2d_layer4(x)
        x = F.relu(self.conv_layer8(x)) 
        x = self.batchnorm2d_layer5(x)
        x = self.dropout2d(x)
        x = x.view(-1, 32*32) #Flattening the conv2D output for dropout 
        x = F.relu(self.fc1(x)) #Fully connected layer with relu activation 
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Parameters
num_outputs = 10 #10 classes
heuristic = True #Choosing from heuristic and Monte Carlo approximation

model = CIFAR10Model(num_outputs)
#model.load_state_dict(torch.load('params_cifar10_dcnn_LR001.ckpt')) #To load a saved model 
print ('\nModel Architecture is:\n', model)
model.cuda() #Sending the model to the GPU
batch_size = 100  #Batch size
loss_func = nn.CrossEntropyLoss() #Cross entropy loss function
if heuristic:
    model.train()
LR = 0.001 #Learning rate 
train_accuracy = []
'''
Train Mode
Create the batch -> Zero the gradients -> Forward Propagation -> Calculating the loss 
-> Backpropagation -> Optimizer updating the parameters -> Prediction 
'''
start_time = time.time()
for epoch in range(150):  # loop over the dataset multiple times
    optimizer = optim.Adam(model.parameters(), lr=LR) #ADAM optimizer
    running_loss = 0.0
    for i, batch in enumerate(data_train, 0):
        data, target = batch
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad() #Zero the gradients at each epoch
        output = model(data)#Forward propagation
        #Negative Log Likelihood Objective function
        loss = loss_func(output, target)
        loss.backward() #Backpropagation 
        optimizer.step() #Updating the parameters using ADAM optimizer
        prediction = output.data.max(1)[1] #Label Prediction 
        accuracy = (float(prediction.eq(target.data).sum())/float(batch_size))*100.0 #Computing the training accuracy
        train_accuracy.append(accuracy)   
    accuracy_epoch = np.mean(train_accuracy)
    print('\nIn epoch ', epoch,' the accuracy of the training set =', accuracy_epoch)
end_time = time.time() 
#torch.save(model.state_dict(), 'params_cifar10_dcnn_LR001.ckpt') #To save the trained model 
'''
Calculate accuracy of trained model on the Test Set
Create the batch ->  Forward Propagation -> Prediction 
'''
correct = 0
total = 0
test_accuracy = []

#Comparing the accuarcy of the heuristic and Monte Carlo Method  
model.eval()
#Disable the droupout layer during test time
#This scales the activation with probability of dropout p at test time 
#So this is applying the heuristic at the test time

#Extra Credit Comparision of Heuristic and Monte Carlo method 
if heuristic:
    MonteCarlo = 1 #Only one itertion if we are using the heuristic 
else:
    MonteCarlo = 100 #100 Monte Carlo iterations for comparision with heuristic method.
    
for batch in data_test:
    data, target = batch
    data, target  = Variable(data).cuda(), Variable(target).cuda()
    for k in range (MonteCarlo):
        if k ==0:
            output = model(data)  #Forward propagation     
        else:
            output+=model(data)  #Forward propagation     
    output/=MonteCarlo #Averaging the softmmax probabilities montecarlo no. of times 
    prediction = output.data.max(1)[1] #Label Prediction
    accuracy = (float(prediction.eq(target.data).sum())/float(batch_size))*100.0 #Computing the test accuracy
    test_accuracy.append(accuracy)
accuracy_test2 = np.mean(test_accuracy)
print('\nAccuracy on the test set = ', accuracy_test2)
