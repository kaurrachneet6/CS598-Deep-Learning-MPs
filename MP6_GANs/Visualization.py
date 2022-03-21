# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:17:32 2018

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
from torch import autograd
from torch.autograd import Variable
import PIL
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

batch_size = 128
n_classes = 10 # 10 Classes in CIFR10

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

    def forward(self, x, extract_features=0): #Specifying the NN architecture 
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
        if(extract_features==4):
            x = F.max_pool2d(x,4,4)
            x = x.view(-1, 196*2*2)
            return x
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
        if(extract_features==8):
            x = F.max_pool2d(x,4,4)
            x = x.view(-1, 196)
            return x
        x = self.LayerNorm_layer8(x) #Layer normalization
        x =  F.leaky_relu(x)
        x = self.pool(x) #Pooling layer
        x = x.view(-1, 196) #Flattening the conv2D output  
        x1 = self.fc1(x) #Fully connected layer with relu activation 
        x10 = self.fc10(x)
        return x1, x10


#Plotting the samples at each epoch to notice the improvement of Generator over time 
def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./CIFARdata', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

''' PART 3: VISUALIZING THE MODELS'''
'''VISUALIZATION PART 1: PERTURB REAL IMGAGES'''
#Selecting a sample batch from the test images and perturb the labels
#Calculating the classification accuracy for the real and fake images in a batch for only discriminator model
#Saving the real, gradients of gradients from alternate classes, fake images

#Loading the discriminator model trained without using the generator
model =  discriminator()
model.load_state_dict(torch.load('discriminator.ckpt')) #To load a saved model 
model.cuda()
model.eval()

batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

#saving the first 100 of the real images
samples = X_batch.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/real_images.png', bbox_inches='tight')
plt.close(fig)

#Taking the output from the Discrimator class generation fc10 layer and classification accuracy
_, output = model(X_batch)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print('Classificication accuracy for only discriminator model on batch of real images is:', accuracy)


#Jittering all the input images
criterion = nn.CrossEntropyLoss(reduce=False)
loss = criterion(output, Y_batch_alternate)

gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                          grad_outputs=torch.ones(loss.size()).cuda(),
                          create_graph=True, retain_graph=False, only_inputs=True)[0]

#Saving the gradients for the jittered images
gradient_image = gradients.data.cpu().numpy()
gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
gradient_image = gradient_image.transpose(0,2,3,1)
fig = plot(gradient_image[0:100])
plt.savefig('visualization/gradient_image.png', bbox_inches='tight')
plt.close(fig)


#Jittering the input images
gradients[gradients>0.0] = 1.0
gradients[gradients<0.0] = -1.0
gain = 8.0
X_batch_modified = X_batch - gain*0.007843137*gradients
X_batch_modified[X_batch_modified>1.0] = 1.0
X_batch_modified[X_batch_modified<-1.0] = -1.0

#Evaluating the accuracy on the new fake images
_, output = model(X_batch_modified)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = (float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print('Classificication accuracy for only discriminator model on batch of fake images is:', accuracy)

#Saving the fake images
samples = X_batch_modified.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)
fig = plot(samples[0:100])
plt.savefig('visualization/jittered_images.png', bbox_inches='tight')
plt.close(fig)


'''VISUALIZAYION PART 2: MAX CLASS OUTPUT'''
#For Discriminator trained without the generator 
#Loading the discriminator model trained without using the generator
model =  discriminator()
model.load_state_dict(torch.load('discriminator.ckpt')) #To load a saved model 
model.cuda()
model.eval()
#Take the mean of 10 copies of one batch of loaded images and assign labels to them
X = X_batch.mean(dim=0)
X = X.repeat(10,1,1,1)

Y = torch.arange(10).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    _, output = model(X)

    loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
    #print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples)
plt.savefig('visualization/max_class_WithoutGenerator.png', bbox_inches='tight')
plt.close(fig)

#For Discriminator trained with the generator 
#Loading the discriminator model trained without using the generator
model2 =  discriminator()
model2.load_state_dict(torch.load('discriminator2.ckpt')) #To load a saved model 
model2.cuda()
model2.eval()

batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

X = X_batch.mean(dim=0)
X = X.repeat(10,1,1,1)

Y = torch.arange(10).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    _, output = model2(X)

    loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
    #print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples)
plt.savefig('visualization/max_class_WithGenerator.png', bbox_inches='tight')
plt.close(fig)

'''VISUALIZATION PART 3: MAX FEATURE OUTPUT'''
#Loading the discriminator model trained without using the generator
model =  discriminator()
model.load_state_dict(torch.load('discriminator.ckpt')) #To load a saved model 
model.cuda()
model.eval()
#Take the mean of 10 copies of one batch of loaded images and assign labels to them
X = X_batch.mean(dim=0)
X = X.repeat(10,1,1,1)

Y = torch.arange(10).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    output = model(X, 4) #For features at layer 4

    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    #print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/max_features_WithoutGenerator_layer4.png', bbox_inches='tight')
plt.close(fig)


#Loading the discriminator model trained with the generator
model2 =  discriminator()
model2.load_state_dict(torch.load('discriminator2.ckpt')) #To load a saved model 
model2.cuda()
model2.eval()
#Take the mean of 10 copies of one batch of loaded images and assign labels to them
X = X_batch.mean(dim=0)
X = X.repeat(10,1,1,1)

Y = torch.arange(10).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    output = model2(X, 4) #For features at layer 4

    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

#saving new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/max_features_WithGenerator_layer4.png', bbox_inches='tight')
plt.close(fig)