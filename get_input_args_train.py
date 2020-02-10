#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PROGRAMMER: Ryan Lin Xiang
# DATE CREATED:   4th Feb 2020                                
# REVISED DATE: 05th Feb 2020
# PURPOSE: This file is used to get the command line arguments parsed for the training file

import argparse


def get_input_args():
    """
    Retrieves and parses the 7 command line arguments provided by the user when
    they run the program from a terminal window.  
    Command Line Arguments:
      1. data_dir : the folder of the training image sets (mandatory)
      2. save_dir : the folder where the trained model should be saved (mandatory)
      3. arch : the architecture of the pre-trained model (mandatory with default)
      4. learning_rate: the rate used to update the weights in the back-propagation (mandatory with default)
      5. hidden_units: number of hidden_units used in the hidden layer of the classifier (mandatory with default)
      6. epochs: number of train iterations of the training image set (mandatory with default)
      7. gpu: boolen showing if GPU should be activated if available (mandatory with default)
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    
    # Creates parse 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', type = str, help = '')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = '')
    parser.add_argument('--learning_rate', type = float, default = '0.0005', help = '')
    parser.add_argument('--hidden_units', type = int, default = '4096', help = '')    
    parser.add_argument('--epochs', type = int, default = '2', help = '')      
    parser.add_argument('--gpu', type = bool, default = '1', help = '')     
    
    return parser.parse_args()   