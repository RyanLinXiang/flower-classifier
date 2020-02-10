#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PROGRAMMER: Ryan Lin Xiang
# DATE CREATED:   4th Feb 2020                                
# REVISED DATE: 05th Feb 2020
# PURPOSE: This file is used to get the command line arguments parsed for the prediction file

import argparse


def get_input_args():
    """
    Retrieves and parses the 5 command line arguments provided by the user when
    they run the program from a terminal window.  
    Command Line Arguments:
      1. image_path: the path to the image file (mandatory)
      2. checkpoint_path : the path to the trained classifier model file (mandatory)
      3. top_k : the number how many top predicted probabilities/category names should be printed out (mandatory with default)
      4. category_names: the file which shows the matches between category indices and names (mandatory with default)
      5. gpu: boolen showing if GPU should be activated if available (mandatory with default)
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates parse 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path')
    parser.add_argument('checkpoint_path')
    parser.add_argument('--top_k', type = int, default = '5', help = 'how many predicted categories should be shown')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'the path to the file which matches category names with indices')  
    parser.add_argument('--gpu', type = bool, default = '1', help = 'should GPU be used for prediction if available?')     
    
    return parser.parse_args()   