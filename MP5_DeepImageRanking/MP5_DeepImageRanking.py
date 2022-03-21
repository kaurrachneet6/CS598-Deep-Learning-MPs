# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 21:05:28 2018

@author: Rachneet Kaur
"""
import numpy as np
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import PIL
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pandas as pd

path = 'Dropbox/tiny-imagenet-200/'
#Function to create a file triplet.txt with as many triplets as images in the training set, 
#their positive image and negative image uniformly sampled
def create_triplets():
    #Creating a .txt file with all the triplets listed in one line 
    train_path = path+'train'
    triplet_file = path+'triplet.txt'
    trainimageclass_file = path +'trainimageclass.txt'
    image_classes = [cl for cl in os.listdir(train_path)] #200 class names 
    triplets = [] #Triplets paths 
    trainimageclass = [] #Images path and labels
    for cl in image_classes:
        images = os.listdir(os.path.join(train_path,cl, 'images')) #All images in the classes
        for image_name in images:
            image_query_name = image_name
            trainimageclass.append(os.path.join(train_path, cl, 'images', image_query_name)+',')
            trainimageclass.append(cl+'\n')
            image_positive_name = np.random.choice(images) 
            #Uniformly sample a positive image until different from query image
            while image_positive_name==image_query_name:
                image_positive_name = np.random.choice(images)
            #Uniformly sample a negative image's class until different from query image's class
            negative_class = np.random.choice(image_classes)
            while negative_class==cl:
                negative_class = np.random.choice(image_classes)
            negative_images = os.listdir(os.path.join(train_path, negative_class, 'images'))
            image_negative_name = np.random.choice(negative_images) 
            triplets.append(os.path.join(train_path, cl, 'images', image_query_name)+',')
            triplets.append(os.path.join(train_path, cl, 'images', image_positive_name)+',')
            triplets.append(os.path.join(train_path, negative_class, 'images', image_negative_name)+'\n')
                
    f = open(triplet_file,'w')
    f.write("".join(triplets))
    f.close()
    
    f1 = open(trainimageclass_file,'w')
    f1.write("".join(trainimageclass))
    f1.close()
        
#Data loader for training set loading a triplet of images each time
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, transform = None, training = 1):
        create_triplets()
        triplet_file = path+'triplet.txt'
        f = open(triplet_file)
        triplets = f.read()
        f.close()
        self.triplet_list=[]
        for line in triplets.split('\n'):
            self.triplet_list.append(line.split(',')) #List of triplets 
        self.triplet_list = self.triplet_list[:-1]
        np.random.shuffle(self.triplet_list)
        
        trainimageclass_file = path+'trainimageclass.txt'
        f1 = open(trainimageclass_file)
        trainimageclass = f1.read()
        f1.close()
        self.trainimageclass_list=[]
        for line in trainimageclass.split('\n'):
            self.trainimageclass_list.append(line.split(',')) #List of triplets 
        self.trainimageclass_list = self.trainimageclass_list[:-1] 
        
        file = open(path+'val/val_annotations.txt')
        file_read = file.read()
        file.close()
        self.val_list=[]
        for line in file_read.split('\n'):
            self.val_list.append(line.split('\t')) #List of annotations with image names 
        self.val_list = self.val_list[:-1]
        
        self.test_path = path+'val/images'
                
        self.transform = transform
        self.training = training
    
    def __getitem__(self, index):
        
        if self.training==1: #For training, return a 
            image_query_name, image_positive_name, image_negative_name = self.triplet_list[index]
            q_image = PIL.Image.open(image_query_name).convert('RGB')
            p_image = PIL.Image.open(image_positive_name).convert('RGB')
            n_image = PIL.Image.open(image_negative_name).convert('RGB')
            
            if self.transform is not None:
                q_image = self.transform(q_image)
                p_image = self.transform(p_image)
                n_image = self.transform(n_image)
    
            triplet = (q_image, p_image, n_image)
    
            return triplet
        
        elif self.training==2:
            image_trained_name, label = self.trainimageclass_list[index]
            trained_image = PIL.Image.open(image_trained_name).convert('RGB')
            if self.transform is not None:
                trained_image = self.transform(trained_image)            
            return (trained_image, label)
            
        elif self.training==3:
            image_test_name = os.path.join(self.test_path, self.val_list[index][0])
            t_label = self.val_list[index][1]
            t_image = PIL.Image.open(image_test_name).convert('RGB')
            if self.transform is not None:
                t_image = self.transform(t_image)
            return (t_image, t_label)

    def __len__(self):
        if self.training==1 or self.training==2:
            return len(self.triplet_list)
        else:
            return len(self.val_list)

#Rescaling the images while loading for pretrained model to 224*224
transformations = transforms.Compose([transforms.Resize(size=(224, 224)),                            
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Training loader after trained
training = 2               
custom_dataset2 = CustomDataset(transformations, training)
train_loader_trained = torch.utils.data.DataLoader(dataset=custom_dataset2, batch_size=1, shuffle=False)

#Testing loader
training = 3
custom_dataset3 = CustomDataset(transformations, training)

test_loader = torch.utils.data.DataLoader(dataset=custom_dataset3, batch_size=1, shuffle=False)

#Parameters
num_outputs = 4096 #4096 feature embedding dimensions


model = torchvision.models.resnet50(pretrained=True) #Loading the pretrained resnet101 model
model.fc = nn.Linear(model.fc.in_features, num_outputs) #4096 feature embedding dimensions

#model.load_state_dict(torch.load('params_deeprank_resnet_pretrained_34.ckpt')) #To load a saved pretrained model 
print ('\nModel Architecture is:\n', model)
model.cuda() #Sending the model to the GPU
batch_size = 20  #Batch size
#Hinge loss for the triplet
loss_func = torch.nn.TripletMarginLoss(margin=1.0, p=2, reduction='elementwise_mean')


                              
model.train()
LR = 0.001 #Learning rate 
train_accuracy = []
epoch_size = 12 #100 epochs training for own model 
    
'''
Train Mode
Create the batch -> Zero the gradients -> Forward Propagation -> Calculating the loss 
-> Backpropagation -> Optimizer updating the parameters -> Prediction 
''' 
start_time = time.time()
loss_epoch = [] #List for losses in each epoch
for epoch in range(epoch_size):  # loop over the dataset multiple times
    #Creates a file triplet.txt with as many triplets as images in the training set, 
    #their positive image and negative image uniformly sampled each time for new epoch
    #Training loader to train  
    training = 1               
    custom_dataset = CustomDataset(transformations, training)
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset, batch_size=20, shuffle=False)#, num_workers = 2)
    training_loss = []
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum = 0.9) #ADAM optimizer
    for i, batch in enumerate(train_loader):
        q_image, p_image, n_image = batch
        q_image, p_image, n_image = Variable(q_image).cuda(), Variable(p_image).cuda(), Variable(n_image).cuda()
        optimizer.zero_grad() #Zero the gradients at each epoch
        q_output = model(q_image)#Forward propagation
        p_output = model(p_image)
        n_output = model(n_image)
        #Negative Log Likelihood Objective function
        loss = loss_func(q_output, p_output, n_output)
        training_loss.append(loss.data)
        if (i%1000==0): 
          print (np.mean(training_loss[i-1000+3: i]))
        torch.save(model.state_dict(), 'params_deeprank_resnet_pretrained_34.ckpt')
        loss.backward() #Backpropagation 
        optimizer.step() #Updating the parameters using ADAM optimizer  
        if (i==4999): 
          torch.save(model.state_dict(), 'params_deeprank_resnet_pretrained_34.ckpt') #To save the trained model
          break
    loss_epoch.append(np.mean(training_loss))
    print('\nIn epoch ', epoch,' the loss of the training set =', loss_epoch)
end_time = time.time() 
print('Time to train the model =', end_time - start_time)
torch.save(model.state_dict(), 'params_deeprank_resnet_pretrained_34.ckpt') #To save the trained model 



#Once the training is complete, plotting the training loss wrt no. of epochs 
plt.figure()
plt.plot(range(1, epoch_size+1), loss_epoch)
plt.title('Training loss wrt epoch')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.savefig('training_loss.jpg')



'''
Calculate accuracy of trained model on the Test Set
Create the batch ->  Forward Propagation -> Prediction 
'''

model.eval()
#Disable the droupout layer during test time  

#Once the Resnet model is trained, calculate the feature embeddings 
#for each training set image and store as a numpy array of size = 100K*4096
trained_embeddings = []
trained_labels =[]
for i, batch in enumerate(train_loader_trained, 0):
    image_trained, label = batch
    image_trained = Variable(image_trained).cuda()
    output = model(image_trained) #Forward propagation
    trained_embeddings.append(output.data)
    trained_labels.append(label) #list of 100K elements with labels for each row of trained_embeddings_array

trained_embeddings_array = []
for i in trained_embeddings:
  trained_embeddings_array.append(np.array(i))
trained_embeddings_array = np.array(trained_embeddings_array)
trained_embeddings_array = trained_embeddings_array[:,0,:]
                                                 
pd.DataFrame(trained_embeddings_array).to_csv('trained_embeddings_array.csv')
pd.DataFrame(trained_labels).to_csv('trained_labels.csv')


#Calculate Precision@30 for testing set
#Once the Resnet model is trained, calculate the feature embeddings 
#for each testing set image and store as a numpy array of size = 10K*4096
test_embeddings = [] 
test_labels =[]
counter = 0
for batch in test_loader:
    data, target = batch
    data  = Variable(data).cuda()
    output = model(data)  #Forward propagation     
    test_embeddings.append(output.data)
    test_labels.append(target) #list of 10K elements with labels for each row of test_embeddings_array
    counter+=1


test_embeddings_array = []
for i in test_embeddings:
  test_embeddings_array.append(np.array(i))
test_embeddings_array = np.array(test_embeddings_array)
test_embeddings_array = test_embeddings_array[:,0,:]
      
pd.DataFrame(test_embeddings_array).to_csv('test_embeddings_array.csv')
pd.DataFrame(test_labels).to_csv('test_labels.csv')
       
#Nearest neighbour classifier to get the 30 nearest embeddings fitted on trained embeddings 
neigh = NearestNeighbors(n_neighbors=30).fit(trained_embeddings_array)

#To test on training set
distances, indices = neigh.kneighbors(trained_embeddings_array)

#To test on training set
avg_precision = []
for i in range(len(trained_embeddings_array[500:600])):
  distances, indices = neigh.kneighbors(trained_embeddings_array[i].reshape(1,-1))

  precision = []
  for j in range(len(indices[0])): #100K iterations 
      #print (j, trained_labels[indices[0][j]][0], trained_labels[i][0])
      precision.append(trained_labels[indices[0][j]][0]==trained_labels[i][0])
  avg_precision.append(np.mean(precision))
  print ('Precision =', i, np.mean(precision))
print('\nPrecision@30 the training set = ', np.mean(avg_precision)*100)
   
#To test on testing set 

avg_precision = []
file = open('Test Precision Results.txt', 'w')
for i in range(len(test_embeddings_array)):
  distances, indices = neigh.kneighbors(test_embeddings_array[i].reshape(1,-1))
  
  precision = []
  for j in range(len(indices[0])): #10K iterations 
      precision.append(trained_labels[indices[0][j]][0]==test_labels[i][0])
  avg_precision.append(np.mean(precision))
  print ('Precision =', i, np.mean(precision))
  file.write(str(precision))
print('\nPrecision@30 the testing set = ', np.mean(avg_precision)*100)
file.close()


#Qualitative Results 
#Choose 5 images from test set and display the top 10 and bottom 10 matches

neighmin10 = NearestNeighbors(n_neighbors=10).fit(trained_embeddings_array)
neighmax10 = NearestNeighbors(n_neighbors=100000).fit(trained_embeddings_array)

for i in range(5):
    test_image, label = custom_dataset3.__getitem__(i)
    plt.figure()
    plt.imshow(test_image.numpy().transpose((1,2,0))) #Image in the test set     
    distance_min10, index_min10 = neighmin10.kneighbors(test_embeddings_array[i].reshape(1,-1))
    print ('Top 10 images:\n')
    counter = 0
    for index in index_min10[0]:
        train_image, label1 = custom_dataset2.__getitem__(index)
        plt.figure()
        plt.imshow(train_image.numpy().transpose((1,2,0)))
        plt.savefig('1.png')
        plt.show()
        plt.close()
        print ('Label and Distance=', label1, distance_min10[0][counter])
        counter+=1
    
    print ('Bottom 10 images:\n')
    distance_all, index_all = neighmax10.kneighbors(test_embeddings_array[i].reshape(1,-1))
    index_max10 = index_all[0][-10:]
    distance_max10 = distance_all[0][-10:]
    counter=0
    for index in index_max10:
        train_image, label2 = custom_dataset2.__getitem__(index)
        plt.imshow(train_image.numpy().transpose((1,2,0)))
        print ('Label and Distance=', label2, distance_max10[counter])
        counter+=1
        
        
        
