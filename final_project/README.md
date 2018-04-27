# Final Project

This project was created and run on the course VM in vagrant so will work easily in that environment, it
was written i python 2.7

It has the following libraries as imports:
import numpy as np
import scipy as sp
import math as m
import scipy.signal 
from scipy.ndimage.filters import gaussian_filter
import cv2

import os
import errno

from os import path
from glob import glob


Input images should be jpg and should be located in the input folder in the root.

The program looks in this folder for the name and runs against it, it will create and output folder named
after the input image and store all stepwise outputs there.

To invoke:

python main.py <image_name>

Example: if image is named dolphin.jpg

python main.py dolphin

