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

#Generator Model
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__() #Specifying the model parameters 
        # input is 3x32x32
        
        #8 convolution layers with ReLU activation 
        self.conv_layer1 = torch.nn.ConvTranspose2d(in_channels = 196, out_channels = 196, kernel_size = 4, stride=2, padding=1)
        '''
        No. of output units in a convolution layer = 
        {[No. of input units - Filter Size + 2(Padding)]/Stride} + 1
        '''
        
        self.conv_layer2 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1) 
        self.conv_layer3 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1) 
        self.conv_layer4 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1) 
        self.conv_layer5 = torch.nn.ConvTranspose2d(in_channels = 196, out_channels = 196, kernel_size = 4, stride=2, padding=1)
        self.conv_layer6 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride = 1, padding = 1)     
        self.conv_layer7 = torch.nn.ConvTranspose2d(in_channels = 196, out_channels = 196, kernel_size = 4, stride=2, padding=1)
        self.conv_layer8 = nn.Conv2d(in_channels=196, out_channels=3, kernel_size=3, stride = 1, padding = 1)  
        
        #1 Fully connected layers with Leaky ReLU activation 
        self.fc1 = nn.Linear(in_features=100, out_features= 196*4*4) #Critic with score between 0 and 1
        
        #Layer Normalization
        self.BatchNorm_layer0 = nn.BatchNorm1d(196*4*4)
        self.BatchNorm_layer1 = nn.BatchNorm2d(196)
        self.BatchNorm_layer2 = nn.BatchNorm2d(196)
        self.BatchNorm_layer3 = nn.BatchNorm2d(196)
        self.BatchNorm_layer4 = nn.BatchNorm2d(196)
        self.BatchNorm_layer5 = nn.BatchNorm2d(196)
        self.BatchNorm_layer6 = nn.BatchNorm2d(196)        
        self.BatchNorm_layer7 = nn.BatchNorm2d(196)  

    def forward(self, x): #Specifying the NN architecture 
        x = self.fc1(x) #Fully connected layer
        x = self.BatchNorm_layer0(x) #Batch normalization for FC layer
        x = x.view(-1, 196, 4, 4)    #Reshaping it to pass through 2D convolution layers 
            
        x = self.conv_layer1(x) #Convolution layers with relu activation
        x = self.BatchNorm_layer1(x) #Batch normalization 
        x = F.relu(x)
        x = self.conv_layer2(x)
        x = self.BatchNorm_layer2(x) #Batch normalization 
        x = F.relu(x)
        x = self.conv_layer3(x)
        x = self.BatchNorm_layer3(x) #Batch normalization 
        x = F.relu(x)
        x = self.conv_layer4(x)
        x = self.BatchNorm_layer4(x) #Batch normalization 
        x = F.relu(x)
        x = self.conv_layer5(x)
        x = self.BatchNorm_layer5(x) #Batch normalization 
        x = F.relu(x)
        x = self.conv_layer6(x)
        x = self.BatchNorm_layer6(x) #Batch normalization)
        x = F.relu(x)
        x = self.conv_layer7(x)
        x = self.BatchNorm_layer7(x) #Batch normalization
        x = F.relu(x)
        x = torch.tanh(self.conv_layer8(x)) #tanh for the last convolution layer
        return x

''' PART 2: Training both the Discriminator and Generator'''
#We need one forward and backward call of both G and D to train G
#We need one forward call of G and two forward and backward calls of D to train D
  
#Calculating the gradient penalty for disciminator norm as in the Wasserstein GAN implementation
def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.cuda()
    
    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

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

#Defining both the models and the optimizers
aD =  discriminator()
#aD.load_state_dict(torch.load('discriminator2.ckpt'))
aD.cuda()

aG = generator()
#aG.load_state_dict(torch.load('generator2.ckpt'))
aG.cuda()

optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))

criterion = nn.CrossEntropyLoss()

#Setting a random noise for the generator using ACGAN implementation
n_z = 100
np.random.seed(352)
label = np.asarray(list(range(10))*10)
noise = np.random.normal(0,1,(100,n_z))
label_onehot = np.zeros((100,n_classes))
label_onehot[np.arange(100), label] = 1
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).cuda()
   
num_epochs = 100
# Train the Generator and Discriminator pair model

gen_train = 1 #Generator is trained each time the discriminator is trained
#If we wish to train discriminator more frequently than the generator, we can set gen_train to say 5

start_time = time.time()

# before epoch training loop starts
loss1 = []
loss2 = []
loss3 = []
loss4 = []
loss5 = []
acc1 = []

for epoch in range(0,num_epochs):
    aG.train()
    aD.train()
    if (epoch%5 ==0):
        torch.save(aD.state_dict(), 'discriminator2.ckpt') #To save the trained model
        torch.save(aG.state_dict(), 'generator2.ckpt') #To save the trained model

    #Included in the training step to prevent overflow error in Adam optimizer
    for group in optimizer_g.param_groups:
        for p in group['params']:
            state = optimizer_g.state[p]
            if('step' in state and state['step']>=1024):
                state['step'] = 1000    
    
    for group in optimizer_d.param_groups:
        for p in group['params']:
            state = optimizer_d.state[p]
            if('step' in state and state['step']>=1024):
                state['step'] = 1000  
    #Training starts batch by batch
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        if(Y_train_batch.shape[0] < batch_size):
            continue   
                     
        # train Generator
        if((batch_idx%gen_train)==0):
            for p in aD.parameters():
                p.requires_grad_(False)
        
            aG.zero_grad()
        
            label = np.random.randint(0,n_classes,batch_size)
            noise = np.random.normal(0,1,(batch_size,n_z))
            label_onehot = np.zeros((batch_size,n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()
        
            fake_data = aG(noise) #Generator generates fake images using the noise
            gen_source, gen_class  = aD(fake_data) 
            #These fake images are sent to the discriminator to decide on label True or fake and the class the generated image may belong to
        
            gen_source = gen_source.mean()
            gen_class = criterion(gen_class, fake_label) 
            #The fake label and label by discriminator is used to calculate cross entropy loss
        
            gen_cost = -gen_source + gen_class
            #Both the losses from the real or fake label and the label of the class label are use to backward propagate
            gen_cost.backward()
            optimizer_g.step()
            #Notice that generator requires one forward and backward pass through both the 
            #generator and the discriminator
       
        # train Discriminator
        for p in aD.parameters():
            p.requires_grad_(True)
        
        aD.zero_grad()
        
        # train discriminator with input from generator
        label = np.random.randint(0,n_classes,batch_size)
        noise = np.random.normal(0,1,(batch_size,n_z))
        label_onehot = np.zeros((batch_size,n_classes))
        label_onehot[np.arange(batch_size), label] = 1
        noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise = Variable(noise).cuda()
        fake_label = Variable(torch.from_numpy(label)).cuda()
        with torch.no_grad():
            fake_data = aG(noise) #Generate fake data from the generator using random noise
        
        disc_fake_source, disc_fake_class = aD(fake_data) 
        #Pass the fake data to discriminiator to generate true or fake label and class label 
        
        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, fake_label)
        #Cross entropy loss for the discriminator generated result
        
        # train discriminator with input from the discriminator
        real_data = Variable(X_train_batch).cuda()
        real_label = Variable(Y_train_batch).cuda()
        
        disc_real_source, disc_real_class = aD(real_data)
        #Training discriminator on the real data 
        
        prediction = disc_real_class.data.max(1)[1]
        accuracy = (float( prediction.eq(real_label.data).sum() ) /float(batch_size))*100.0
        
        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, real_label)
        #Cross entropy loss for the discriminator generated labels on the real data set
        
        gradient_penalty = calc_gradient_penalty(aD,real_data,fake_data)
        # Applying the gradient penalty to the discriminator
        
        # within the training loop
        loss1.append(gradient_penalty.item())
        loss2.append(disc_fake_source.item())
        loss3.append(disc_real_source.item())
        loss4.append(disc_real_class.item())
        loss5.append(disc_fake_class.item())
        acc1.append(accuracy)
        
        if((batch_idx%50)==0):
            print(epoch, batch_idx, "%.2f" % np.mean(loss1), 
                                    "%.2f" % np.mean(loss2), 
                                    "%.2f" % np.mean(loss3), 
                                    "%.2f" % np.mean(loss4), 
                                    "%.2f" % np.mean(loss5), 
                                    "%.2f" % np.mean(acc1))
        
        disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
        #Calculating the total loss based on the discriminator trained on the real data and trained on fake data generated by the generator
        disc_cost.backward()
        optimizer_d.step() 
    print('\nIn epoch ', epoch,' the accuracy of the training set =', np.mean(acc1))
    #Note that training of the discroiminator requires one forward call of the 
    #generator and two (one for real data and another for fake data) forward and backward calls of the discriminator
    
    #Testing the Generator Discriminator pair trained at each epoch
    aD.eval()
    with torch.no_grad():
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch, Y_test_batch= Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()
    
            with torch.no_grad():
                _, output = aD(X_test_batch)
    
            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
            test_accu.append(accuracy)
            accuracy_test = np.mean(test_accu)
    print('\nIn epoch ', epoch, 'the testing accuracy is ', accuracy_test)
    print ('Time to train this epoch of model is:', time.time()-start_time)

    #Saving the output after every epoch
    with torch.no_grad():
        aG.eval()
        samples = aG(save_noise)
        samples = samples.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0,2,3,1)
        aG.train()
    
    fig = plot(samples)
    plt.savefig('output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
    plt.close(fig)
    
    if(((epoch+1)%1)==0):
        torch.save(aG,'tempG.model')
        torch.save(aD,'tempD.model')   

torch.save(aG.state_dict(), 'generator2.ckpt')
torch.save(aD.state_dict(), 'discriminator2.ckpt')
       