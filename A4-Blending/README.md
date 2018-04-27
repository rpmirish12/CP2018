# Pyramid Blending

## Synopsis

Putting together a pyramid blending pipeline that allows combination of separate images into a seamlessly blended image. The technique is based on the paper [“A multiresolution spline with application to image mosaics”](http://persci.mit.edu/pub_pdfs/spline83.pdf) (Burt and Adelson; ACM 1983) 

![Pyramid Blending](blend.png)

## Instructions

- Images in the `images/source/sample` directory are provided for testing 

- Execute the blending pipeline by running `python main.py`. The script will look inside each subfolder under `images/source`, looking for folders that have images with filenames that end with 'white', 'black' and 'mask'. For each such folder it finds, it will apply the blending procedure to them, and save the output to a folder with the same name as the input in `images/output/`. (For example, `images/source/sample` will produce output in `images/output/sample`.)

- The blending procedure splits the input images into their blue, green, and red channels and blends each channel separately. 

- Along with the output blended image, main.py will create visualizations of the Gaussian and Laplacian pyramids for the blend. 


### 1. Implement the functions in the `blending.py` file.

  - `reduce_layer`: Blur and subsample an input image
  - `expand_layer`: Upsample and blur an input image
  - `gaussPyramid`: Construct a gaussian pyramid from the image by repeatedly reducing the input
  - `laplPyramid`: Construct a laplacian pyramid by taking the difference between gaussian layers
  - `blend`: Combine two laplacian pyramids through a weighted sum
  - `collapse`: Flatten a blended pyramid into a final image

The docstrings of each function contains detailed instructions. 
