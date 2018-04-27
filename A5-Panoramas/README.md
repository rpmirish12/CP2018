# Panoramas

## Synopsis

Align & stitch together a series of images into a panorama. (Szeliski chapter 9.1).


## Instructions

- Script to execute panorama pipeline by running `python main.py`. The script will look inside each subfolder under `images/source` and attempt to apply the panorama procedure to the contents, and save the output to a folder with the same name as the input in `images/output/`. (For example, `images/source/sample` will produce output in `images/output/sample`.)


### 1. Implement the functions in the `panorama.py` file.

  - `getImageCorners`: Return the x, y coordinates for the four corners of an input image
  - `findHomography`: Return the transformation between the keypoints of two images
  - `getBoundingCorners`: Find the extent of a canvas large enough to hold two images
  - `warpCanvas`: Warp an input image to align with the next adjacent image
  - `blendImagePair`: Fit two images onto a single canvas

The docstrings of each function contains detailed instructions. 
