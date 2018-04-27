""" Pyramid Blending
This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.
References
----------
See the following papers, available on T-square under references:
(1) "The Laplacian Pyramid as a Compact Image Code"
        Burt and Adelson, 1983
(2) "A Multiresolution Spline with Application to Image Mosaics"
        Burt and Adelson, 1983
Notes
-----
    You may not use cv2.pyrUp or cv2.pyrDown anywhere in this assignment.
GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but these functions should NOT save the image to disk.
    2. DO NOT import any other libraries aside from those that we provide.
    You should be able to complete the assignment with the given libraries
    (and in many cases without them).
    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.
    4. This file has only been tested in the course virtual environment.
    You are responsible for ensuring that your code executes properly in the
    virtual machine environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""
import numpy as np
import scipy as sp
import scipy.signal  # one option for a 2D convolution library
import cv2


def generatingKernel(a):
    """Return a 5x5 generating kernel based on an input parameter (i.e., a
    square "5-tap" filter.)
    Parameters
    ----------
    a : float
        The kernel generating parameter in the range [0, 1] used to generate a
        5-tap filter kernel.
    Returns
    -------
    output : numpy.ndarray
        A 5x5 array containing the generated kernel
    """
    # DO NOT CHANGE THE CODE IN THIS FUNCTION
    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)


def reduce_layer(image, kernel=generatingKernel(0.4)):
    """Convolve the input image with a generating kernel and then reduce its
    width and height each by a factor of two.
    For grading purposes, it is important that you use a reflected border
    (i.e., padding equivalent to cv2.BORDER_REFLECT101) and only keep the valid
    region (i.e., the convolution operation should return an image of the same
    shape as the input) for the convolution. Subsampling must include the first
    row and column, skip the second, etc.
    Example (assuming 3-tap filter and 1-pixel padding; 5-tap is analogous):
                          fefghg
        abcd     Pad      babcdc   Convolve   ZYXW   Subsample   ZX
        efgh   ------->   fefghg   -------->  VUTS   -------->   RP
        ijkl    BORDER    jijklk     keep     RQPO               JH
        mnop   REFLECT    nmnopo     valid    NMLK
        qrst              rqrsts              JIHG
                          nmnopo
    A "3-tap" filter means a 3-element kernel; a "5-tap" filter has 5 elements.
    Please consult the lectures for a more in-depth discussion of how to
    tackle the reduce function.
    Parameters
    ----------
    image : numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data type
        (e.g., np.uint8, np.float64, etc.)
    kernel : numpy.ndarray (Optional)
        A kernel of shape (N, N). The array may have any data type (e.g.,
        np.uint8, np.float64, etc.)
    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        An image of shape (ceil(r/2), ceil(c/2)). For instance, if the input is
        5x7, the output will be 3x4.
    """
    # re-format input to float64 for output generation, try and catch if fail
    try:
        image = np.float_(image) # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
    except ValueError:
        print('Input array not able to be converted to float64 leveraging np.float64()')

    # Convolve with padding of zeroes for upsampled output    
    # reduce_output = scipy.signal.convolve2d(image, kernel, 'same', boundary='fill', fillvalue=0) # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html 
    reduce_output = sp.ndimage.filters.convolve(image, kernel, mode = 'mirror') # https://piazza.com/class/jc55krnjbsgm9?cid=435
    # subsample skipping every other row and column
    reduce_output = reduce_output[::2, ::2] # https://stackoverflow.com/questions/9027862/what-does-listxy-do 

    return np.float_(reduce_output)


def expand_layer(image, kernel=generatingKernel(0.4)):
    """Upsample the image to double the row and column dimensions, and then
    convolve it with a generating kernel.
    Upsampling the image means that every other row and every other column will
    have a value of zero (which is why we apply the convolution after). For
    grading purposes, it is important that you use a reflected border (i.e.,
    padding equivalent to cv2.BORDER_REFLECT101) and only keep the valid region
    (i.e., the convolution operation should return an image of the same
    shape as the input) for the convolution.
    Finally, multiply your output image by a factor of 4 in order to scale it
    back up. If you do not do this (and you should try it out without that)
    you will see that your images darken as you apply the convolution.
    You must explain why this happens in your submission PDF.
    Example (assuming 3-tap filter and 1-pixel padding; 5-tap is analogous):
                                          000000
             Upsample   A0B0     Pad      0A0B0B   Convolve   zyxw
        AB   ------->   0000   ------->   000000   ------->   vuts
        CD              C0D0    BORDER    0C0D0D     keep     rqpo
        EF              0000   REFLECT    000000    valid     nmlk
                        E0F0              0E0F0F              jihg
                        0000              000000              fedc
                                          0E0F00
                NOTE: Remember to multiply the output by 4.
    A "3-tap" filter means a 3-element kernel; a "5-tap" filter has 5 elements.
    Please consult the lectures for a more in-depth discussion of how to
    tackle the expand function.
    Parameters
    ----------
    image : numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)
    kernel : numpy.ndarray (Optional)
        A kernel of shape (N, N). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)
    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        An image of shape (2*r, 2*c). For instance, if the input is 3x4, then
        the output will be 6x8.
    """
    # re-format input to float64 for output generation, try and catch if fail
    try:
        image = np.float_(image) # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
    except ValueError:
        print('Input array not able to be converted to float64 leveraging np.float64()')
    
    # Take the rows and columns of the image and double them for expanded output
    (i_rows, i_cols) = image.shape[:2]
    upsample = np.zeros((i_rows*2, i_cols*2))

    # Add image inputs to every other row and column
    upsample[::2, ::2] = image # https://stackoverflow.com/questions/9027862/what-does-listxy-do 

    # Convolve with padding of zeroes for upsampled output
    #expand_output =  scipy.signal.convolve2d(upsample, kernel, 'same', boundary='fill', fillvalue=0) # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
    expand_output = sp.ndimage.filters.convolve(upsample, kernel, mode = 'mirror') # https://piazza.com/class/jc55krnjbsgm9?cid=435
    # scale image intensity back up
    expand_output = expand_output * 4 

    return np.float_(expand_output)


def gaussPyramid(image, levels):
    """Construct a pyramid from the image by reducing it by the number of
    levels specified by the input.
    You must use your reduce_layer() function to generate the pyramid.
    Parameters
    ----------
    image : numpy.ndarray
        An image of dimension (r, c).
    levels : int
        A positive integer that specifies the number of reductions to perform.
        For example, levels=0 should return a list containing just the input
        image; levels = 1 should perform one reduction and return a list with
        two images. In general, len(output) = levels + 1.
    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A list of arrays of dtype np.float. The first element of the list
        (output[0]) is layer 0 of the pyramid (the image itself). output[1] is
        layer 1 of the pyramid (image reduced once), etc.
    """
    # re-format input to float64 for output generation, try and catch if fail
    try:
        image = np.float_(image) # https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html
    except ValueError:
        print('Input array not able to be converted to float64 leveraging np.float64()')
    
    # ensure levels are set higher than 0
    try:
        levels > 0
    except ValueError:
        print("gaussPyramid: Levels value must be greater than 0")

    # instantiate list of outputs with image in position 0
    gauss_output = [] 
    gauss_output.append(image)# https://stackoverflow.com/questions/37747021/create-numpy-array-of-images

    # iterate through the defined number of levels recursively running the reduce_layer function for each output, and append to list
    for x in xrange(levels):    # https://www.pythoncentral.io/how-to-use-pythons-xrange-and-range/
        gauss_output.append(reduce_layer(gauss_output[x]))

    return gauss_output


def laplPyramid(gaussPyr):
    """Construct a Laplacian pyramid from a Gaussian pyramid; the constructed
    pyramid will have the same number of levels as the input.
    You must use your expand_layer() function to generate the pyramid. The
    Gaussian Pyramid that is passed in is the output of your gaussPyramid
    function.
    Parameters
    ----------
    gaussPyr : list<numpy.ndarray(dtype=np.float)>
        A Gaussian Pyramid (as returned by your gaussPyramid function), which
        is a list of numpy.ndarray items.
    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of the same size as gaussPyr. This pyramid should
        be represented in the same way as guassPyr, as a list of arrays. Every
        element of the list now corresponds to a layer of the laplacian
        pyramid, containing the difference between two layers of the gaussian
        pyramid.
        NOTE: The last element of output should be identical to the last layer
              of the input pyramid since it cannot be subtracted anymore.
    Notes
    -----
        (1) Sometimes the size of the expanded image will be larger than the
        given layer. You should crop the expanded image to match in shape with
        the given layer. If you do not do this, you will get a 'ValueError:
        operands could not be broadcast together' because you can't subtract
        differently sized matrices.
        For example, if my layer is of size 5x7, reducing and expanding will
        result in an image of size 6x8. In this case, crop the expanded layer
        to 5x7.
    """
    laplacian = []
    iterations = len(gaussPyr)-1
    for x in xrange(iterations): # One less due to the last layer not being able to subtracted

        # Expand the gaussian images to be subtracted from original and store the dimensions
        expansion = expand_layer(gaussPyr[x+1])
        (expand_rows, expand_cols) = expansion.shape[:2]
        (gauss_rows, gauss_cols) = gaussPyr[x].shape[:2]

        # If the dimensions of the expansion is too large, crop to smaller image value
        if expand_rows != gauss_rows:
            expand_rows = gauss_rows
        if expand_cols != gauss_cols:
            expand_cols = gauss_cols
        
        # Reset the adjustments and cropping if set for the expaned image
        expansion = expansion[:expand_rows, :expand_cols]

        # Append the generated difference of the gaussian from its expansion to generate the laplacian
        laplacian.append(gaussPyr[x]-expansion)

    # Append the final image to the end of the laplacian pyramid
    laplacian.append(gaussPyr[-1])
    return laplacian


def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
    """Blend two laplacian pyramids by weighting them with a gaussian mask.
    You should return a laplacian pyramid that is of the same dimensions as the
    input pyramids. Every layer should be an alpha blend of the corresponding
    layers of the input pyramids, weighted by the gaussian mask.
    Therefore, pixels where current_mask == 1 should be taken completely from
    the white image, and pixels where current_mask == 0 should be taken
    completely from the black image.
    (The variables `current_mask`, `white_image`, and `black_image` refer to
    the images from each layer of the pyramids. This computation must be
    performed for every layer of the pyramid.)
    Parameters
    ----------
    laplPyrWhite : list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of an image constructed by your laplPyramid
        function.
    laplPyrBlack : list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of another image constructed by your laplPyramid
        function.
    gaussPyrMask : list<numpy.ndarray(dtype=np.float)>
        A gaussian pyramid of the mask. Each value should be in the range
        [0, 1].
    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A list containing the blended layers of the two laplacian pyramids
    Notes
    -----
        (1) The input pyramids will always have the same number of levels.
        Furthermore, each layer is guaranteed to have the same shape as
        previous levels.
    """
    Blend = []

    # Iterated through the total range, since they're equal, selected the white lapl
    for x in range(len(laplPyrWhite)):
        #Blend function from lecture and paper, multiply the gaussian mask by the white laplacian, and the 0 value multiplier of the black laplacian
        Blend.append(gaussPyrMask[x]*laplPyrWhite[x] + (1-gaussPyrMask[x])*laplPyrBlack[x])

    return Blend


def collapse(pyramid):
    """Collapse an input pyramid.
    Approach this problem as follows: start at the smallest layer of the
    pyramid (at the end of the pyramid list). Expand the smallest layer and
    add it to the second to smallest layer. Then, expand the second to
    smallest layer, and continue the process until you are at the largest
    image. This is your result.
    Parameters
    ----------
    pyramid : list<numpy.ndarray(dtype=np.float)>
        A list of numpy.ndarray images. You can assume the input is taken
        from blend() or laplPyramid().
    Returns
    -------
    numpy.ndarray(dtype=np.float)
        An image of the same shape as the base layer of the pyramid.
    Notes
    -----
        (1) Sometimes expand will return an image that is larger than the next
        layer. In this case, you should crop the expanded image down to the
        size of the next layer. Look into numpy slicing to do this easily.
        For example, expanding a layer of size 3x4 will result in an image of
        size 6x8. If the next layer is of size 5x7, crop the expanded image
        to size 5x7.
    """
    # Set the iterator to start at the end of the list
    last_value = len(pyramid)-1

    # Initialize the first value of the pyramid for the recursive call, initially missed this, thanks piazza!
    collapsed_out = pyramid[-1] # https://piazza.com/class/jc55krnjbsgm9?cid=404 

    for x in xrange(last_value,0,-1): # https://stackoverflow.com/questions/3476732/how-to-loop-backwards-in-python/3476768 
        # Use same logic as the laplacian pyramid function for checks and crops
        expansion = expand_layer(collapsed_out)
        (expand_rows, expand_cols) = expansion.shape[:2]

        # Iterate in reverse for next value of pyramid to compare
        (pyramid_rows, pyramid_cols) = pyramid[x-1].shape[:2]
        if expand_rows != pyramid_rows:
            expand_rows = pyramid_rows
        if expand_cols != pyramid_cols:
            expand_cols = pyramid_cols
        expansion = expansion[:expand_rows, :expand_cols]

        # recursively add the expansion values to generate the new collapsed_out
        collapsed_out = expansion + pyramid[x-1]
    return np.float_(collapsed_out)