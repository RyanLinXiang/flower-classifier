#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PROGRAMMER: Ryan Lin Xiang
# DATE CREATED:   4th Feb 2020                                
# REVISED DATE: 05th Feb 2020
# PURPOSE: This file included all the helper functions necessary to save and load the model, as well as process the images

from PIL import Image
import numpy as np
from torch import save, load, from_numpy
import torchvision.models as models



def save_model(model, save_dir, class_to_idx, arch, structure):
    """
    Saves the trained and validated model
    Parameters:
     model : the trained and validated model
     save_dir : the path where the classifier model file should be saved
     class_to_idx : the path to the file where category indices are saved to trace back the indiced predicted by the model
     arch : the architecture of the pre-trained model chosen
     structure : the structure of the classifier used to initiate the model
    Returns:
     None
    """
    
    classifier = {'arch': arch,
                  'class_to_idx': class_to_idx, 
                  'state_dict': model.classifier.state_dict(),
                  'structure': structure}
    
    save(classifier, save_dir+"/"+"classifier.pth")
    
    
    
def load_model(classifier_path):
    """
    Load the pre-trained model from the specified file with the updated classifier (features are frozen)
    Parameters:
     classifier_path : the path to the saved classifier model
    Returns:
        classifier model
    """    

    classifier = load(classifier_path)
    
    model = getattr(models, classifier['arch'])
    model = model(pretrained=True)
    
    model.classifier = classifier['structure']
    
    model.class_to_idx = classifier['class_to_idx']
       
    model.classifier.load_state_dict(classifier['state_dict'])
 
    return model



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    Parameters:
        image : the path to the image file
    Returns:
        numpy array with the image file processed and ready as input for the model        
    '''
   
    img = Image.open(image)    
    
    # the shortest dimension of the image gets width of 256 while the other dimension is resized with respect to the ratio
    if img.size[0] < img.size[1]:
        ratio = 256/img.size[0]
        img = img.resize((256,int(img.size[1]*ratio)))
    else:
        ratio = 256/img.size[1]
        img = img.resize((int(img.size[0]*ratio),256))

    # crop a square of 224px from the center of the image in order to get the image ready for the model
    top = (img.size[1] - 224)/2
    bottom = (img.size[1] + 224)/2
    left = (img.size[0] - 224)/2
    right = (img.size[0] + 224)/2    
    img = img.crop((left, top, right, bottom))
        
    img = np.array(img)/255
    
    # normalization of the image in order to get the image ready for the model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])  
    img = np.transpose((img - mean) / std)

    return from_numpy(img)    

