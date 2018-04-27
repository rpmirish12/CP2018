#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""This is a program for seam carving of images of various sizes.

It was written for the GT OMSCS program to adhere to the requirements of an 
assignment in the course.

Use of this program requires python 2.7. To run the program place images you 
wish to adjust in the /images directory as .png files. The CLI command is as
follows:
        
        python main.py <image_namge> <pixel_count_to_adjust>

        -image_name = name of the .png file WITHOUT .png
        -pixel_count_to_adjust = number of pixels to add/remove as an integer

Author: Ryan Miller
Date: 2/28/2018
"""


import cv2
import numpy as np

import os
import errno

from os import path
from glob import glob

from seams1 import seam_carve

import sys


def main(image, image_name, pixels):
    """ Function that calls seams1.py and passes parameters.
   
    input: <image.png> <string name> <int pixels>
    output: no return output
    """
    output = seam_carve(image, image_name, pixels) # call main function in seams1.py and return removal image
    removal_name = image_name + "_seams_removed.png" # set image name
    cv2.imwrite(removal_name, output.astype(np.uint8)) # write image to file for seam removal


if __name__ == "__main__":
    image_name = sys.argv[1] # set image name to second value in cli
    input_image = "images/"+image_name+".png" # adjust the path of the image
    
    try: # check to see if the image can be read in, if not, respond back
        seam_cut_image = cv2.imread(input_image, cv2.IMREAD_COLOR).astype(np.float_)
    except OSError:
        print("cannot read in image", sys.argv[1])
    
    try: # check to see if the pixel count can be converted to an integer
        pixels = np.int_(sys.argv[2])
    except ValueError:
        print("vertical_pixels not able to be an integer", sys.argv[2])

    main(seam_cut_image, image_name, pixels) # call main function to begin processing
