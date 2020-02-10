#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PROGRAMMER: Ryan Lin Xiang
# DATE CREATED:   4th Feb 2020                                
# REVISED DATE: 05th Feb 2020
# PURPOSE: This file is used to train and valid a pretrained classifier model. It can be performed through the command line.
#          The training and validation status progress is shown with every batch number processed and every 50 batches statistics
#          are printed out

from get_input_args_train import get_input_args
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np
import os.path
from os import path
from utilities import save_model


def main():
    """
    This is the main program
    """
    
    # Initiating variables with parsed command line arguments
    
    in_arg = get_input_args()
 
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test' 
    save_dir = in_arg.save_dir
    arch = in_arg.arch.lower()
    epochs = in_arg.epochs
    hidden_units = in_arg.hidden_units
    learning_rate = in_arg.learning_rate
    gpu = in_arg.gpu
    
    
    # Correct the variables if necessary to avoid incorrect calculations
    # Collect error messages what variables have been changed to what values
    
    error_messages = []
    
    if (epochs <= 0):
        epochs = 1
        error_messages.append("epochs was corrected to 1")
    elif (epochs > 10):
        epochs = 10
        error_messages.append("epochs was corrected to 10")   
        
    if (learning_rate <= 0.000001):
        learning_rate = 0.00001
        error_messages.append("learning_rate was corrected to 0.00001")
    elif (learning_rate >= 0.1):
        learning_rate = 0.01
        error_messages.append("learning_rate was corrected to 0.01") 
        
    if (hidden_units < 4):
        hidden_units = 4
        error_messages.append("hidden_units was corrected to 4")
    
          
    if not save_dir:
        save_dir = os.getcwd()
        save = False
    elif save_dir == "/": # slash means that the new trained classified should be stored in the current directory
        save = True
        save_dir = os.getcwd()
    else:
        save = True
        
    
    if path.exists(data_dir) and path.exists(train_dir) and path.exists(valid_dir) \
    and path.exists(test_dir) and path.exists(save_dir): # check if all paths are correct
        
        if (arch in "alexnet,vgg16,densenet161"): # check if the stated architecture is supported
                                 
            # define the data transforms of the train data  
            data_transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                                        transforms.RandomResizedCrop(224),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
            
            # define the data transforms of the validation data
            data_transforms_valid = transforms.Compose([transforms.Resize(256),
                                                        transforms.CenterCrop(224),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
            
            # load the image data of the train and validation set and perform transforms
            train_data = datasets.ImageFolder(train_dir, transform=data_transforms_train)
            valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms_valid)

        
            # load the transformed image data into loader variables by batches
            trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
            validloader = DataLoader(valid_data, batch_size=64, shuffle=True)
 
        
            # download the pretrained version of the selected model defined by the parser variable 'arch'
            model = getattr(models, arch)
            model = model(pretrained=True)
     
            
            # freeze the parameters of the model as only classifier will be updated
            for param in model.parameters():
                param.requires_grad = False
                
            
            # classifier layer will be fully connected so get the input units first
            if (arch == "vgg16"):
                num_in_features = model.classifier[0].in_features   
            elif (arch == "densenet161"):    
                num_in_features = model.classifier.in_features 
            elif (arch == "alexnet"):
                num_in_features = model.classifier[1].in_features 

                
            # define the new classifier und replace the current one    
            new_classifier = nn.Sequential(nn.Linear(num_in_features, hidden_units),
                                           nn.ReLU(),
                                           nn.Dropout(0.5),
                                           nn.Linear(hidden_units, int(hidden_units/4)),
                                           nn.ReLU(),
                                           nn.Dropout(0.5),
                                           nn.Linear(int(hidden_units/4), 102),
                                           nn.LogSoftmax(dim=1))
    
            model.classifier = new_classifier

            # use GPU power if available and model to it
            if gpu:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = "cpu"

            model.to(device)

            # define the loss function and the optimizer
            criterion = nn.NLLLoss()
            optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)


            print_every = 50
            train_len = len(trainloader)
            valid_len = len(validloader)

            # start training and validation epochs
            for epoch in range(epochs):
                epoch += 1
                last_print = 0
                running_loss = 0
    
                for batch, (inputs, labels) in enumerate(trainloader):       
                    batch += 1
            
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # for each batch gradients should be zeroed
                    optimizer.zero_grad()
        
                    # perform feed-forward and calculate loss through back propagation
                    logps = model.forward(inputs)
                    loss = criterion(logps, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
       
                    print (f"Epoch {epoch}/{epochs}, batch {batch}/{train_len}")
        
                    # do validation on the test set after defined set of batches

                    if (((batch) % print_every == 0) & (batch >= print_every)) or (batch == train_len):
                        valid_loss = 0
                        accuracy = 0
                        
                        # put into evaluation mode and deactivate gradients for feed-forward validation 
                        model.eval()                    
                        with torch.no_grad():
                            # iterate through the valid data set
                            for inputs, labels in validloader:
                                inputs, labels = inputs.to(device), labels.to(device)
                                logps = model.forward(inputs)
                                
                                # calculate the losses
                                batch_loss = criterion(logps, labels)
                                valid_loss += batch_loss.item()
                    
                                # calculate accuracy and return the category with the top probability
                                # then compare with the labels and calculate the mean of the right matches
                                ps = torch.exp(logps)
                                top_p, top_class = ps.topk(1, dim=1)
                                equals = top_class == labels.view(*top_class.shape)
                                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                           
                        # after each train and validation circle defined by the number of batches before print the statistics
                        if (batch == train_len): 
                            print(f"Train loss: {running_loss/(train_len - last_print):.3f}.. ", end = '')
                        else:
                            last_print += print_every # print statistics after each epoch
                            print(f"Train loss: {running_loss/print_every:.3f}.. ", end = '')
                
                        print(f"Valid loss: {valid_loss/valid_len:.3f}.. ", end = '')
            
                        print(f"Accuracy: {accuracy/valid_len:.3f}")
                        running_loss = 0

                        # switch back to train mode
                        model.train()
        
            # if save path is defined
            if (save == True):
                save_model(model, save_dir, train_data.class_to_idx, arch, new_classifier)
                    
        else:
            print ("Architecture chosen not supported!")
            
        
    else:
        print ("Incorrect directories - please check!")


    # print out error messages if any
    if (len(error_messages)):
        for v in error_messages:
            print (v)        
        
        
if __name__ == "__main__":
    main()