# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 21:05:28 2018

@author: Rachneet Kaur
"""

import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import PIL
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

batch_size = 128

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./CIFARdata', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./CIFARdata', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

#Discriminator Model
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__() #Specifying the model parameters 
        # input is 3x32x32
        
        #8 convolution layers with ReLU activation 
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=196, kernel_size=3, stride = 1, padding = 1) 
        '''
        No. of output units in a convolution layer = 
        {[No. of input units - Filter Size + 2(Padding)]/Stride} + 1
        '''
        
        self.conv_layer2 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 2, padding = 1) 
        self.conv_layer3 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1) 
        self.conv_layer4 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 2, padding = 1) 
        self.conv_layer5 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1) 
        self.conv_layer6 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1)     
        self.conv_layer7 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1) 
        self.conv_layer8 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 2, padding = 1)  
        
        #2 Fully connected layers with Leaky ReLU activation 
        self.fc1 = nn.Linear(in_features=196, out_features= 1) #Critic with score between 0 and 1
        #Image size is 1*1 with 196 channels from the last convolution layer
        
        self.fc10 = nn.Linear(in_features=196, out_features= 10) #Auxillary classifier for class description
        
        #Layer Normalization
        self.LayerNorm_layer1 = nn.LayerNorm([196, 32, 32])
        self.LayerNorm_layer2 = nn.LayerNorm([196, 16, 16]) 
        self.LayerNorm_layer3 = nn.LayerNorm([196, 16, 16]) 
        self.LayerNorm_layer4 = nn.LayerNorm([196, 8, 8])
        self.LayerNorm_layer5 = nn.LayerNorm([196, 8, 8]) 
        self.LayerNorm_layer6 = nn.LayerNorm([196, 8, 8]) 
        self.LayerNorm_layer7 = nn.LayerNorm([196, 8, 8])
        self.LayerNorm_layer8 = nn.LayerNorm([196, 4, 4])  
        
        #Max pooling
        self.pool = nn.MaxPool2d(kernel_size = 4, stride=4, padding = 0) #Max pooling 

    def forward(self, x): #Specifying the NN architecture 
        x = self.conv_layer1(x) #Convolution layers with Leaky relu activation
        x = self.LayerNorm_layer1(x) #Layer normalization 
        x =  F.leaky_relu(x)
        x = self.conv_layer2(x)
        x = self.LayerNorm_layer2(x) #Layer normalization 
        x =  F.leaky_relu(x)
        x = self.conv_layer3(x)
        x = self.LayerNorm_layer3(x) #Layer normalization 
        x =  F.leaky_relu(x)
        x = self.conv_layer4(x)
        x = self.LayerNorm_layer4(x) #Layer normalization 
        x =  F.leaky_relu(x)
        x = self.conv_layer5(x)
        x = self.LayerNorm_layer5(x) #Layer normalization 
        x =  F.leaky_relu(x)
        x = self.conv_layer6(x)
        x = self.LayerNorm_layer6(x) #Layer normalization)
        x =  F.leaky_relu(x)
        x = self.conv_layer7(x)
        x = self.LayerNorm_layer7(x) #Layer normalization
        x =  F.leaky_relu(x)
        x = self.conv_layer8(x)
        x = self.LayerNorm_layer8(x) #Layer normalization
        x =  F.leaky_relu(x)
        x = self.pool(x) #Pooling layer
        x = x.view(-1, 196) #Flattening the conv2D output  
        x1 = self.fc1(x) #Fully connected layer with relu activation 
        x10 = self.fc10(x)
        return x1, x10
    
''' PART 1: Training only the Discriminator'''  
model =  discriminator()
model.load_state_dict(torch.load('discriminator.ckpt')) #To load a saved model 
model.cuda()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001
model.train()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
train_accuracy = []

for epoch in range(5):  # loop over the dataset multiple times
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0 
    #Included in the training step to prevent overflow error in Adam optimizer
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            if('step' in state and state['step']>=1024):
                state['step'] = 1000
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
    
        if(Y_train_batch.shape[0] < batch_size):
            continue
    
        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        _, output = model(X_train_batch) #Only 10 class output for Discriminator only training 
    
        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()
    
        loss.backward()
        optimizer.step()
        prediction = output.data.max(1)[1] #Label Prediction 
        accuracy = (float(prediction.eq(Y_train_batch.data).sum())/float(batch_size))*100.0 #Computing the training accuracy
        train_accuracy.append(accuracy)   
    accuracy_epoch = np.mean(train_accuracy)
    print('\nIn epoch ', epoch,' the accuracy of the training set =', accuracy_epoch)
    if (epoch%5==0):
        torch.save(model.state_dict(), 'discriminator.ckpt') #To save the trained model 

'''
Calculate accuracy of trained model on the Test Set
Create the batch ->  Forward Propagation -> Prediction 
'''
correct = 0
total = 0
test_accuracy = []
model.eval()

for batch in testloader:
    data, target = batch
    data, target  = Variable(data).cuda(), Variable(target).cuda()
    _, output = model(data)  #Forward propagation     
    prediction = output.data.max(1)[1] #Label Prediction
    accuracy = (float(prediction.eq(target.data).sum())/float(batch_size))*100.0 #Computing the test accuracy
    test_accuracy.append(accuracy)
accuracy_test2 = np.mean(test_accuracy)
print('\nAccuracy on the test set = ', accuracy_test2)

   