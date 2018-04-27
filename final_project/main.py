#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Ryan Miller
Date: 04/22/2018
"""


import cv2
import numpy as np

import os
import errno

from os import path
from glob import glob

from toon import toonify

import sys
SRC_FOLDER = "images/input"
OUT_FOLDER = "images/output"
EXTENSIONS = set(["bmp", "jpeg", "jpg", "png", "tif", "tiff"])


def main(image, name):
    """ Function that calls tooni.py and passes parameters.
   
    input: <image.png>
    output: no return output
    """

    toonify(image, name)
    return


if __name__ == "__main__":
    image_name = sys.argv[1]
    input_image = "images/input/"+image_name+".jpg"

    try:
        name = sys.argv[1]
    except ValueError:
        print("invalid name", sys.argv[1])
    try:
        image_f = cv2.imread(input_image, cv2.IMREAD_COLOR)
    except OSError:
        print("cannot read image", sys.argv[1])

    directory = OUT_FOLDER+"/"+name
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    main(image_f, directory)

