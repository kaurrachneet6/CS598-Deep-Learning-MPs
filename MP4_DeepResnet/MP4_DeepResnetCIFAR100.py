# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:05:28 2018

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
import torch.utils.model_zoo as model_zoo

#Boolean to decide if we are using the pretrained network (True) or not(False)
pretrained = False

#No rescaling the images if own model is used
if not pretrained:
    #Transforming data and normalizing the CIFAR100 testing dataset
    transforms_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    #Defining the data augmentation transformations for the training set
    transformations = transforms.Compose([transforms.RandomHorizontalFlip(p=0.2),
                                          transforms.RandomVerticalFlip(p=0.2),                             
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Rescaling the images while loading for pretrained model to 224*224
if pretrained:
    #Transforming data and normalizing the CIFAR100 testing dataset
    transforms_test = transforms.Compose(
        [transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    #Defining the data augmentation transformations for the training set
    transformations = transforms.Compose([transforms.Resize(size=(224, 224)),
                                          transforms.RandomHorizontalFlip(p=0.2),
                                          #transforms.RandomVerticalFlip(p=0.2),                             
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Training set 
train = torchvision.datasets.CIFAR100(root='./CIFARdata', train=True,
                                        download=True, transform=transformations)
data_train = torch.utils.data.DataLoader(train, batch_size=100, #Batch size = 100
                                          shuffle=True, num_workers=0)

#Testing set
test = torchvision.datasets.CIFAR100(root='./CIFARdata', train=False,
                                       download=True, transform=transforms_test)
data_test = torch.utils.data.DataLoader(test, batch_size=100, #Batch size = 100
                                         shuffle=False, num_workers=0)

#Model URL for pretrained RESNET-18
model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}

#Implementing a basic block 
#Variables are: Filter dimensions, input and output channels, stride and padding 
class BasicBlock(nn.Module):
    def __init__(self, filter_x, filter_y, input_channels, output_channels, stride_dim, padding_dim):
        super(BasicBlock, self).__init__()
        self.match_needed = False
        self.conv_block_layer1 = nn.Conv2d(input_channels, output_channels, (filter_x, filter_y), stride = stride_dim, padding = padding_dim)
        self.batchnorm_block_layer1 = nn.BatchNorm2d(output_channels)
        if (stride_dim>1):
            self.match_needed = True
        self.match_dim = nn.Conv2d(input_channels, output_channels, kernel_size = 1, stride = stride_dim) 
        #To balance the downsample in the size of the image while conv_layer with stride = 2
        self.conv_block_layer2 = nn.Conv2d(output_channels, output_channels, (filter_x, filter_y), stride = 1, padding = padding_dim)
        self.batchnorm_block_layer2 = nn.BatchNorm2d(output_channels)
        

    def forward(self, x):
        res = x
        if (self.match_needed):
            res =  self.match_dim(res) #To balance the downsample in size of the images
        x_new = self.conv_block_layer1(x) #Convolution
        x_new = self.batchnorm_block_layer1(x_new) #Batchnorm
        x_new = F.relu(x_new) #ReLU
        x_new = self.conv_block_layer2(x_new) #Convolution
        x_new = self.batchnorm_block_layer2(x_new) #Batchnorm
        x_new += res #Adding to the residual 
        return x_new

#Implementing the Resnet architecture with Basic Block 
class RESNETModel(nn.Module):
    def __init__(self, b_block, num_outputs): #Basic Block and Num of output classes
        super(RESNETModel, self).__init__() #Specifying the model parameters 
        #Input image is 3x32x32
        
        #convolution layer
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride = 1, padding = 1) 
        '''
        No. of output units in a convolution layer = 
        {[No. of input units - Filter Size + 2(Padding)]/Stride} + 1 = 
        (32 - 3 + 2(1))/1 + 1 = 32 
        '''
        self.basic_layer1 = b_block(3, 3, 32, 32, 1, 1)
        self.basic_layer2 = b_block(3, 3, 32, 32, 1, 1)
        self.conv_layer_upsample1 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size = 1) 
        #To balance the channel doubling change 
        self.basic_layer3 = b_block(3, 3, 64, 64, 2, 1) 
        #Only the first Conv. layer has stride = 2, rest all have stride = 1
        #Size of the image halves
        self.basic_layer4 = b_block(3, 3, 64, 64, 1, 1)        
        self.basic_layer5 = b_block(3, 3, 64, 64, 1, 1)
        self.basic_layer6 = b_block(3, 3, 64, 64, 1, 1) 
        self.conv_layer_upsample2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size = 1) 
        #To balance the channel doubling change 
        self.basic_layer7 = b_block(3, 3, 128, 128, 2, 1)
        #Size of the image halves further
        self.basic_layer8 = b_block(3, 3, 128, 128, 1, 1)        
        self.basic_layer9 = b_block(3, 3, 128, 128, 1, 1)
        self.basic_layer10 = b_block(3, 3, 128, 128, 1, 1) 
        self.conv_layer_upsample3 = nn.Conv2d(in_channels=128, out_channels=256,kernel_size = 1) 
        #To balance the channel doubling change 
        self.basic_layer11 = b_block(3, 3, 256, 256, 2, 1)
        #Size of the image halves further
        self.basic_layer12 = b_block(3, 3, 256, 256, 1, 1)         
        #Fully connected layers with ReLU activation 
        self.fc1 = nn.Linear(in_features=256, out_features= num_outputs) #Fully connected layer at the end      
        
        #Dropout 
        self.dropout2d = nn.Dropout2d(p=0.3)  # p - Probability of dropping out a neuron
        self.batchnorm2d_layer1 = nn.BatchNorm2d(32) #Batch Normalization
        self.pool = nn.MaxPool2d(4, stride=4) #Max pooling 

    def forward(self, x): #Specifying the NN architecture 
        x = F.relu(self.conv_layer1(x)) #Convolution layers with relu activation
        x = self.batchnorm2d_layer1(x) #Batch normalization 
        x = self.dropout2d(x) #Dropout
        x = self.basic_layer1(x)
        x = self.basic_layer2(x)
        x = self.conv_layer_upsample1(x) #To balance the channel doubling change 
        x = self.basic_layer3(x)
        x = self.basic_layer4(x)
        x = self.basic_layer5(x)
        x = self.basic_layer6(x)
        x = self.conv_layer_upsample2(x) #To balance the channel doubling change 
        x = self.basic_layer7(x)
        x = self.basic_layer8(x)
        x = self.basic_layer9(x)
        x = self.basic_layer10(x)
        x = self.conv_layer_upsample3(x) #To balance the channel doubling change 
        x = self.basic_layer11(x)
        x = self.basic_layer12(x)
        x = self.pool(x) #Pooling layer
        x = x.view(x.size(0), -1) #Flattening the conv2D output for dropout 
        x = self.fc1(x) #Fully connected layer
        return x


#Parameters
num_outputs = 100 #100 classes in CIFAR100
basic_block = BasicBlock

#Defining the model with own architecture if pretrained = False
if not pretrained:
    model = RESNETModel(basic_block, num_outputs)

#Loading the model weights 
if pretrained:
    model = torchvision.models.resnet18(pretrained=True) #Loading the pretrained resnet18 model #Fine tuning 
    model.fc = nn.Linear(model.fc.in_features, num_outputs) # 100 classes in CIFAR100

#model.load_state_dict(torch.load('params_cifar100_RESNET_LR001.ckpt')) #To load a saved own model 
#model.load_state_dict(torch.load('params_cifar100_resnet_LR001_pretrained.ckpt')) #To load a saved pretrained model 
print ('\nModel Architecture is:\n', model)
model.cuda() #Sending the model to the GPU
batch_size = 100  #Batch size
loss_func = nn.CrossEntropyLoss() #Cross entropy loss function
model.train()

if pretrained:
  LR = 0.01 #Learning rate #0.01 for pretrained model
else:
  LR = 0.001
train_accuracy = []

if not pretrained:
    epoch_size = 50 #100 epochs training for own model 
else:
    epoch_size = 5 #50 epochs training for pretrained model
    
'''
Train Mode
Create the batch -> Zero the gradients -> Forward Propagation -> Calculating the loss 
-> Backpropagation -> Optimizer updating the parameters -> Prediction 
'''

start_time = time.time()
for epoch in range(epoch_size):  # loop over the dataset multiple times
    if (epoch>30):
      LR = 0.0001
    if pretrained:
      optimizer = optim.SGD(model.parameters(), lr=LR) #SGD optimizer
    else:
      optimizer = optim.Adam(model.parameters(), lr=LR) #ADAM optimizer
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
print('Time to train the model =', end_time - start_time)
#torch.save(model.state_dict(), 'params_cifar100_RESNET_LR001.ckpt') #To save the trained own model 
#torch.save(model.state_dict(), 'params_cifar100_resnet_LR001_pretrained.ckpt') #To save the trained pretrained version of model 

'''
Calculate accuracy of trained model on the Test Set
Create the batch ->  Forward Propagation -> Prediction 
'''

correct = 0
total = 0
test_accuracy = []

model.eval()
#Disable the droupout layer during test time   
for batch in data_test:
    data, target = batch
    data, target  = Variable(data).cuda(), Variable(target).cuda()
    output = model(data)  #Forward propagation     
    prediction = output.data.max(1)[1] #Label Prediction
    accuracy = (float(prediction.eq(target.data).sum())/float(batch_size))*100.0 #Computing the test accuracy
    test_accuracy.append(accuracy)
print('\nAccuracy on the test set = ', np.mean(test_accuracy))
