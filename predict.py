#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PROGRAMMER: Ryan Lin Xiang
# DATE CREATED:   4th Feb 2020                                
# REVISED DATE: 05th Feb 2020
# PURPOSE: This file is used to classify the probable category of a flower image passed to it through command line. It shows 
#          the image label based on the folder name and prints out the top category probabilities 

from get_input_args_predict import get_input_args
import torch
import numpy as np
from os import path
from utilities import load_model, process_image
import json

# this is the main program
def main():
    in_arg = get_input_args()
  
    # initiate the variables passed through command line
    image_path = in_arg.image_path
    checkpoint_path = in_arg.checkpoint_path
    top_k = in_arg.top_k
    category_names = in_arg.category_names
    gpu = in_arg.gpu


    # Correct the variables if necessary to avoid incorrect calculations
    # Collect error messages what variables have been changed to what values    
    error_messages = []
    
    if (top_k <= 0):
        top_k = 1
        error_messages.append("top_k was corrected to 1")
    elif (top_k > 5):
        top_k = 5
        error_messages.append("top_k was corrected to 5")   
        
    
    if path.isfile(image_path) and path.isfile(checkpoint_path) and path.isfile(category_names): # check if all files are existing
        
        # load the categoy names file which connects category indices with indices predicted by the model 
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)

        # use the folder of the specified file as category index
        title_idx = image_path.split("/")[-2]
     
        # find the name by matching the category index with the indices in the category names file
        img_label = [v for k, v in cat_to_name.items() if k == title_idx]
        img_label = img_label[0] 
        print (f"Image label: {img_label}")
            
        # use GPU power if available for prediction  
        if gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = "cpu"        
        
        # load the model from the specified classifier path
        model = load_model(checkpoint_path)
        model = model.to(device)
        
        # freeze all model parameters to save speed and put model
        for param in model.parameters():
            param.requires_grad = False
            
        # switch to evaluation mode
        model.eval()
        
        # prepare the specified image so that it can be used by the model
        img = process_image(image_path)   
        img = img[None, :, :, :]
        img = img.float()
        img = img.to(device)

        # deactive all gradients to further speed up the performance and calculate the log prob outputs
        with torch.no_grad():
            logps = model.forward(img)
       
        # convert to probability values
        ps = torch.exp(logps)
 
        # save the top probabilities and their category indices
        top_p, top_class = ps.topk(top_k, dim=1)       
        top_p = np.array(top_p).reshape(top_k)
        top_class = np.array(top_class)
    
        # find the the category indices for the predicted top indices
        top_classes = [k for k, v in model.class_to_idx.items() if v in top_class]
        # match the top category indices with their category names
        names = [cat_to_name[v] for v in top_classes if v in cat_to_name]    
        
        for i, name in enumerate(names):
                print (f"{name}: ..... {format(top_p[i],'.2f')}")
        

    else:
        print ("Incorrect paths to files - please check!")
        
    # print out error messages if any    
    if (len(error_messages)):
        for v in error_messages:
            print (v)
  
        
if __name__ == "__main__":
    main()