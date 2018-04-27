import numpy as np
import scipy as sp
import math as m
import scipy.signal 
from scipy.ndimage.filters import gaussian_filter
import cv2

OUT_FOLDER = ""
OUT_NAME = ""

def generatingKernel():
    """Generates approximate kernel for use in custom edge detection,
    with the returning output yielding the sobel x and y 3x3 kernels
    approximating the derivative of a Gaussian.

    Parameters
    ----------
    None

    Returns
    -------
    output : 2 numpy.ndarray
        numpy.ndarray(dtype=np.float64)
        3x3 s_x Gaussian kernel

        numpy.ndarray(dtype=np.float64)
        3x3 s_y Gaussian kernel
    """

    # https://www.quora.com/What-is-the-difference-between-edge-detection-Sobel-detection-and-Canny-detection
    # https://en.wikipedia.org/wiki/Sobel_operator
    # Sobel operator x  
    kernel_x = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ])

    # Sobel operator y
    kernel_y = np.array([
        [1,2,1],
        [0,0,0],
        [-1,-2,-1]
    ])

    # return sobel x and sobel y
    return kernel_x, kernel_y


def cv_gray(image):
    """Converts a BGR image to gray for edge detection
    algoithm.

    Parameters
    ----------
    numpy.ndarray
        A BGR image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    Returns
    -------
    numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)
    """

    # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
    # CV2 color convert
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # write to file
    out = OUT_FOLDER+"/gray.jpg"
    cv2.imwrite(out, image)

    # return single channel image
    return image


def median_filter(image):
    """Takes an image input and smooths it applying a median filter
    reducing the salt and pepper noise in the image. Following the 
    white paper a 7x7 size kernel was used.

    Parameters
    ----------
    numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    Returns
    -------
    numpy.ndarray
        A grayscale image of shape (r, c).

    """

    # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=medianblur#medianblur
    # run the image through a median blur with 7x7 kernel convolution
    image = cv2.medianBlur(image, 7)

    # write output to disk
    out = OUT_FOLDER+"/median.jpg"
    cv2.imwrite(out, image)

    # return smoothed image
    return image


def canny_edge(image, weight=0.13):
    """Takes an image input and applies canny-edge detection
    to it yielding single pixel edges. The reference paper set
    the weight to 0.33 but through some testing I found 0.13 worked
    better for images I used.

    Returns an image composed of the identified edges.

    Parameters
    ----------
    numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)
    
    weight : float
        A decimale weight value to be used as to select upper and lower bounds
        of the edge detection algorithm.

    Returns
    -------
    numpy.ndarray
        A grayscale image of shape (r, c).

    """

    # detect upper and lower bounds by first taking the average
    # intensity of the image pixel values
    median = np.median(image)

    # set the upper value to the floor of a weighted multiplier
    lower = m.floor(max(0, (1.0 - weight) * median))
    
    # set the lower value to the floor of a weighted multiplier
    upper = m.floor(min(255, (1.0 + weight) * median))

    # run cv2.canny operation on image with upper and lower bounds
    edge = cv2.Canny(image, lower, upper)

    # write output to file
    out = OUT_FOLDER+"/edge.jpg"
    cv2.imwrite(out, edge)

    # return edge values
    return edge


def custom_edge_gaussian(image, ksize=5.0):
    """Takes an image input and applies a gaussian filter
    to it to smooth the image and remove noise with an
    kernel size of 7x7.

    Returns smoothed image.

    Following wiki for reference:
    https://en.wikipedia.org/wiki/Canny_edge_detector

    Parameters
    ----------
    numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)
    
    kszie : float
        Kernel size for the gaussian convolution

    Returns
    -------
    numpy.ndarray
        A grayscale image of shape (r, c).

    """

    # https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
    # run a gaussian filter of the image with a 7x7 kernel
    output = gaussian_filter(image, ksize)

    # write the output file
    out = OUT_FOLDER+"/gaussian.jpg"
    cv2.imwrite(out, output)

    # return the smooth image
    return output


def gradient_intesity(image):
    """ Calculate the gradient intensity filtering in all 3 directions
    about the pixel: vertical, horizontal, diagonal in a blurred image
    using the sobel operator.

    Returns the intensity matrix of an image's gradient values and the
    directions.

    Following wiki for reference:
    https://en.wikipedia.org/wiki/Canny_edge_detector

    Parameters
    ----------
    numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    Returns
    -------
    intensity: numpy.ndarray
        A grayscale image of shape (r, c).
    
    direction: numpy.ndarray
        Array of theta values calculated using atan2

    """

    # set sobel operators
    kernel_x, kernel_y = generatingKernel()

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
    # convolve to get gradient magnitude sobel_x
    intensity_x = scipy.signal.convolve2d(image, kernel_x)

    # convolve to get gradient magnitude sobel_y
    intensity_y = scipy.signal.convolve2d(image, kernel_y)
    
    # resize the arrays as convolution adds 2 rows and 2 cols
    # remove top and bottom output rows
    intensity_x = np.delete(intensity_x, (0), axis=0)
    intensity_y = np.delete(intensity_y, (0), axis=0)
    intensity_x = np.delete(intensity_x, (intensity_x.shape[0]-1), axis=0)
    intensity_y = np.delete(intensity_y, (intensity_y.shape[0]-1), axis=0)

    # remove top and bottom output cols
    intensity_x = np.delete(intensity_x, (0), axis=1)
    intensity_y = np.delete(intensity_y, (0), axis=1)
    intensity_x = np.delete(intensity_x, (intensity_x.shape[0]-1), axis=1)
    intensity_y = np.delete(intensity_y, (intensity_y.shape[0]-1), axis=1)

    # write the output to file for x and y gradient magnitudes.
    out = OUT_FOLDER+"/grad_intensityx.jpg"
    cv2.imwrite(out, intensity_x)
    out = OUT_FOLDER+"/grad_intensityy.jpg"
    cv2.imwrite(out, intensity_x)
    
    # calculate the overall gradient using the hypot function
    intensity = np.hypot(intensity_x, intensity_y)

    # calculate the direction of the gradient
    direction = np.arctan2(intensity_y, intensity_x)

    # write the gradient magnitude output to file
    out = OUT_FOLDER+"/grad_intensity.jpg" 
    cv2.imwrite(out, intensity)
    
    # return matrix of gradient magnitudes and array of directions
    return intensity, direction


def angle_buckets(theta):
    """For canny algorithm the edge direction angle is rounded to one of 
    four angles representing vertical, horizontal and the two 
    diagonals (0, 45, 90 and 135)

    Function takes the magnitude directions and places them in the appropriate
    bucket.

    Following wiki for reference:
    https://en.wikipedia.org/wiki/Canny_edge_detector

    Parameters
    ----------
    theta : numpy.ndarray
        Array of degree values in radians that corresponds to the edge
        locations in the image.

    Returns
    -------
    theta : numpy.ndarray
        array of adjusted degree buckets alligned to the appropriate direction:
        (0, 45, 90 and 135)
    """

    # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.rad2deg.html
    # convert the radian measurements to degrees
    theta = np.rad2deg(theta)

    # adjust the value over 180 for those higher than 180
    theta = theta % 180

    # following: https://en.wikipedia.org/wiki/Canny_edge_detector
    # first bucket is the two extreme ends at 0
    if (0 <= theta <= 22.5) or (157.5 <= theta <= 180):
        theta = 0

    # second bucket is +- 22.5 from 45
    elif (22.5 <= theta < 67.5):
        theta = 45
    
    # third bucket is +- 22.5 from 90
    elif (67.5 <= theta < 112.5):
        theta = 90

    # final bucket is +- 22.5 from 135
    elif (112.5 <= theta < 157.5):
        theta = 135

    # return array of adjusted theta values
    return theta


def non_max_suppression(image, direction):
    """Non-maximum suppression is an edge thinning technique. Goal is to
    get only 1 accurate resoponse for an edge. The algorithm for each pixel 
    in the gradient image is:

        Compare the edge strength of the current pixel with the edge strength 
        of the pixel in the positive and negative gradient directions. If the 
        edge strength of the current pixel is the largest compared to the 
        other pixels in the mask with the same direction (i.e., the pixel 
        that is pointing in the y-direction, it will be compared to the pixel 
        above and below it in the vertical axis), the value will be preserved. 
        Otherwise, the value will be suppressed.

    Function takes the image edge matrix and array of directions and returns
    single pixel edges.

    Following wiki for reference:
    https://en.wikipedia.org/wiki/Canny_edge_detector

    Parameters
    ----------
    image : numpy.ndarray
        matrix of edge values for an image

    direction : numpy.ndarray
        array of directional theta values matched to gradient magnitudes of
        edges.

    Returns
    -------
    output : numpy.ndarray
        edge matrix of single pixel edges
    """

    # get the height and width of the image
    height, width = image.shape[:2]

    # generate the output matrix of zeros
    output = np.zeros((height, width))

    # iterate through the rows and cols of the edge matrix and
    # compare to all neighboring pixels to determine if the value
    # will be preserved or suppressed, if not set in loop, will 
    # be 0
    for row in xrange(1,height-1):
        for col in xrange(1,width-1):
            # get the direction value at the edge position
            theta = angle_buckets(direction[row, col])

            # check if 0 degree bucket
            if theta == 0:
                # for 0 degrees the point will be considered to be on the edge 
                # if its gradient magnitude is greater than the magnitudes at pixels 
                # in the east and west directions
                if (image[row,col] >= image[row, col-1]):
                    if (image[row,col] >= image[row, col+1]):
                        output[row,col] = image[row,col]
            
            # check if 90 degree bucket
            elif theta == 90:
                # for 90 degrees the point will be considered to be on the edge if its 
                # gradient magnitude is greater than the magnitudes at pixels in the 
                # north and south directions
                if (image[row,col] >= image[row-1, col]):
                    if (image[row,col] >= image[row+1, col]):
                        output[row,col] = image[row,col]

            # check if 135 degree bucket        
            elif theta == 135:
                # for 135 degrees the point will be considered to be on the edge if its 
                # gradient magnitude is greater than the magnitudes at pixels in the 
                # north west and south-east directions
                if (image[row,col] >= image[row-1, col-1]):
                    if (image[row,col] >= image[row+1, col+1]):
                        output[row,col] = image[row,col]

            # check if 45 degree bucket    
            elif theta == 45:
                 # for 45 degrees the point will be considered to be on the edge if its 
                 # gradient magnitude is greater than the magnitudes at pixels in the 
                 # north east and south west directions
                if (image[row,col] >= image[row-1, col+1]):
                    if (image[row,col] >= image[row+1, col-1]):
                        output[row,col] = image[row,col]
    
    # write the output to file
    out = OUT_FOLDER+"/suppressed.jpg"
    cv2.imwrite(out, output)

    # return the edge matrix
    return output


def double_threshold(image, upper, lower):
    """Filter out edge pixels with a weak gradient value and preserve edge 
    pixels with a high gradient value. This is accomplished by selecting 
    high and low threshold values. If an edge pixels gradient value is 
    higher than the high threshold value, it is marked as a strong edge 
    pixel. If an edge pixels gradient value is smaller than the high 
    threshold value and larger than the low threshold value, it is marked as 
    a weak edge pixel. If an edge pixels value is smaller than the low 
    threshold value, it will be suppressed.

    Function takes the image edge matrix and upper and lower gradient
    magnitude values and returns lists of strong and weak indexes
    and a new suppressed edge output.

    Following wiki for reference:
    https://en.wikipedia.org/wiki/Canny_edge_detector

    Parameters
    ----------
    image : numpy.ndarray
        matrix of edge values for an image

    upper : int_
        upper bound of the gradient magnitude value
    
    lower : int_
        lower bound of the gradient magnitude value

    Returns
    -------
    image : numpy.ndarray
        edge matrix of pixel edges
    
    weak : numpy.ndarray
        indeces of strong pixel edges

    strong : numpy.ndarray
        indeces of weak pixel edges
    """

    # create lists for pixel identification
    strong = []
    weak = []
    suppressed = []

    # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.where.html 
    # get the index values where the edge is greater than the upper bound
    strong = np.where(image > upper)

    # get the index values where the edge is greater than the lower bound but
    # less than the upper bound
    weak = np.where((image >= lower) & (image < upper))

    # get the index values where the edge is lower than the lower bound
    suppressed = np.where(image < lower)

    # set the suppressed index values to 0
    image[suppressed[0], suppressed[1]] = 0

    # set the weak values to lower bound of 45
    image[weak[0], weak[1]] = 45

    # set the weak values to upper bound of 45
    image[strong[0], strong[1]] = 255

    # write output to file
    out = OUT_FOLDER+"/threshold.jpg"
    cv2.imwrite(out, image)

    # return the matrix of edges, and indexes of strong and weak edges
    return image, weak, strong


def edge_track(image, weak, strong=255):
    """Determine weak pixel values at edges and remove them, these pixels 
    can either be extracted from the true edge, or the noise/color variations.

    To track take a pixel and compare it to the 8 values surrounding it, if
    any of the neighboring pixels are strong, then keep it, otherwise suppress
    it setting it's magnitude to 0.

    Following wiki for reference:
    https://en.wikipedia.org/wiki/Canny_edge_detector

    Parameters
    ----------
    image : numpy.ndarray
        matrix of edge values for an image

    weak : list <int>
        upper bound indexes
    
    strong : list <int>
        lower bound indexes
        
    Returns
    -------
    image : numpy.ndarray
        edge matrix of pixel edges
    
    """

    # get the height and width of the image.
    (height, width) = image.shape[:2]
    
    # iterate through the edges, if the pixel value
    # equals the weak pixel ID: 45, then check all neighboring pixels
    # if one is strong set the pixel to strong, otherwise suppress it
    for row in xrange(height):
        for col in xrange(width):

            # check to see if weak pixel
            if image[row, col] == 45:

                # check if pixel to right is strong
                if (image[row+1,col] == strong):
                    image[row][col] = strong

                # check if pixel to upper right is strong
                elif (image[row+1,col-1] == strong):
                    image[row][col] = strong

                # check if pixel to lower right is strong
                elif (image[row+1,col+1] == strong):
                    image[row][col] = strong
                
                # check if pixel to left is strong
                elif (image[row-1,col] == strong):
                    image[row][col] = strong
                
                # check if pixel to bottom left is strong
                elif (image[row-1,col+1] == strong):
                    image[row][col] = strong
                
                # check if pixel to upper left is strong
                elif (image[row-1,col-1] == strong):
                    image[row][col] = strong
                
                # check if pixel below is strong
                elif (image[row,col+1] == strong):
                    image[row][col] = strong
                
                # check if pixel above is strong
                elif (image[row,col-1] == strong):
                    image[row][col] = strong
                
                # if no strong pixels around, suppress
                else:
                    image[row][col] = 0

    # write output to file
    out = OUT_FOLDER+"/custom_edge.jpg"
    cv2.imwrite(out, image)

    # return edge matrix
    return image


    


def edge_dilation(edge):
    """Morphological operation that takes the edges and leverge a 2x2 kernel 
    increases there width. Purpose of this is to bolden and smooth the lines
    for the artistic shading affect.
    
    Parameters
    ----------
    edge : numpy.ndarray
        matrix of edge values for an image

    Returns
    -------
    dilation : numpy.ndarray
        adjusted edge matrix of pixel edges
    
    """

    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html 
    # set the kernel to a 2x2 set of ones following the reference paper
    kernel = np.ones((2,2), np.uint8)

    # calculate dilation with kernel on a single iteration
    dilation = cv2.dilate(edge, kernel, iterations=1)

    # write the output to file
    out = OUT_FOLDER+"/edge_dilated.jpg"
    cv2.imwrite(out, dilation)

    # return the dilated edges
    return dilation


def threshold_func(dilate):
    """Filter the edges post dilation so that small  contours
    picked up by the Canny edge detector are ignored in the final
    image are ignored. Reduces unwanted clutter.

    Followed reference paper with min value set to 10.
    
    Parameters
    ----------
    dilate : numpy.ndarray
        matrix of edge values for an image

    Returns
    -------
    output : numpy.ndarray
        adjusted edge matrix of pixel edges
    
    """

    # set dilatation matrix to unit8 for threshold function
    dilate = dilate.astype(np.uint8)

    # https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
    # suppress values below 10 to remove unwanted clutter, invert image
    # so edges are now black
    ret, output = cv2.threshold(dilate, 10, 255, cv2.THRESH_BINARY_INV)

    # write output to file
    out = OUT_FOLDER+"/threshold2.jpg"
    cv2.imwrite(out, output)

    # return output edge matrix
    return output


def downsample(image):
    """Downsample the image for bi-lateral filter, helps in performance.
    Following the reference paper downsize by a factor of 4.
    
    Parameters
    ----------
    image : numpy.ndarray
        matrix of pixel values for an image

    Returns
    -------
    output : numpy.ndarray
        adjusted image matrix of pixels downsized by 4x
    
    """

    # set height/width of input image
    height, width = image.shape[:2]

    # calculate new height and width
    height = int(height/4)
    width = int(width/4)

    # set the new dimensions
    dim = (width, height)

    # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
    # resize the image downsampling to 1/4 size
    downsampled = cv2.resize(image, dim, interpolation = cv2.INTER_LANCZOS4)

    # write output to file
    out = OUT_FOLDER+"/downsampled.jpg"
    cv2.imwrite(out, downsampled)

    # return downsampled image
    return downsampled

def bi_filter(image, iterations=14, ksize=9):
    """Calculate the output of a bilateral filter where weights of pixel
    values are further adjusted depending on how different they are. This
    assists in maintaining the sharpness of the image as high changes in 
    color value are kept more distinct (achieving part of the cartoon affect).

    Functions runs the filter following the refernce paper: 14 iterations and with
    a kernel size of 9.
    
    Parameters
    ----------
    image : numpy.ndarray
        matrix of pixel values for an image
    
    iterations : int
        number times to run the filter

    ksize : int
        size of the kernel for the filter

    Returns
    -------
    output : numpy.ndarray
        adjusted image matrix of pixels 
    
    """

    # https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
    # run bilateral filter over the image
    bi = cv2.bilateralFilter(image, ksize, 9, 7)

    # iterate based on defined runtimes in default case, 14 times
    for i in xrange(iterations-1):
        bi = cv2.bilateralFilter(bi, ksize, 9, 7)

    # write output to file
    out = OUT_FOLDER+"/bi_downsampled.jpg"
    cv2.imwrite(out, bi)

    # return the filtered image
    return bi


def upsample(image, height, width):
    """Post filtering, need to upsample image back to its original 
    size. This function uses INTER_LANCZ0S4 interpolation for added
    smoothness in final output.
    
    Parameters
    ----------
    image : numpy.ndarray
        matrix of pixel values for an image
    
    height : int
        height of original image

    width : int
        width of orignal image

    Returns
    -------
    output : numpy.ndarray
        adjusted image matrix of pixels upsampled to orignal size
    
    """

    # set the dimensions to original image size
    dim = (width, height)

    # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
    # resize the image upsampling to original size
    upsampled = cv2.resize(image, dim, interpolation=cv2.INTER_LANCZOS4)

    # write output to file
    out = OUT_FOLDER+"/bi_upsampled.jpg"
    cv2.imwrite(out, upsampled)

    # return upsampled image
    return upsampled


def median_filter_upsample(image):
    """Takes an image input and smooths it applying a median filter
    reducing the salt and pepper noise in the image. Following the 
    white paper a 7x7 size kernel was used.

    Parameters
    ----------
    numpy.ndarray
        A image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    Returns
    -------
    numpy.ndarray
        A image of shape (r, c).

    """

    # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=medianblur#medianblur
    # run the image through a median blur with 7x7 kernel convolution
    image = cv2.medianBlur(image, 7)

    # write output to disk
    out = OUT_FOLDER+"/median_upsample.jpg"
    cv2.imwrite(out, image)

    # return image
    return image


def quantize_colors(image, height, width, factor=24):
    """Quantize the colors to reduce the number of distinct colors
    in the image. This is done to produce a simpler overall color
    palette making the image look more hand colored than generated.

    functions takes in an image and resets it pixel values on BGR
    channels to a scaled floor based on a multiply factor chosen
    on input. Default is set to the reference paper value of 24.

    Parameters
    ----------
    numpy.ndarray
        A image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    height : int
        height of the output image
    
    width : int
        width of the output image

    factor : int
        scaling factor, default set to 24

    Returns
    -------
    numpy.ndarray
        A image of shape (r, c).

    """

    # create the the output matrix
    output = np.zeros((height, width, 3))

    # iterate through the image and re-calculate the BGR channel values
    # based on scaling factor
    for i in xrange(height):
        for j in xrange(width):

            # set the output of BGR channels to the floor divider value
            # multiplied by the scaling factor
            output[i][j][0] = m.floor(image[i][j][0]/factor) * factor
            output[i][j][1] = m.floor(image[i][j][1]/factor) * factor
            output[i][j][2] = m.floor(image[i][j][2]/factor) * factor
    
    # write output to file
    out = OUT_FOLDER+"/quantized.jpg"
    cv2.imwrite(out, output)

    # return output
    return output


def combine(image, edges, height, width):
    """Combine the edges matrix with the color adjusted image
    to effectively draw lines over distinct features, toon-ing
    the overall picture.

    Parameters
    ----------
    image : numpy.ndarray
        A image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    edges : numpy.ndarray
        matrix of edges
    
    height : int
        width of the output image

    width : int
        scaling factor, default set to 24

    Returns
    -------
    numpy.ndarray
        A image of shape (r, c).

    """

    # set np array for final image values
    final_image = np.zeros((height,width,3))

    # loop through the image and combine the edge values
    # with the image values
    for i in xrange(height):
        for j in xrange(width):

            # if an edge exists store it, otherwise set image value
            if(edges[i][j] == 0):
                final_image[i,j,:] = 0
            else:
                final_image[i,j] = image[i,j]
    
    # write output to file
    out = OUT_FOLDER+"/final.jpg"
    cv2.imwrite(out, final_image)


def toonify(image, directory):
    """Main driver of toonify. Takes an input image and runs it through
    the CP toonify pipeline:
        - convert to grayscale
        - median filter
        - edge detection
        - morphological operations
        - edge filter
        - bilateral filter
        - median filter
        - quantize colors
        - recombine

    Parameters
    ----------
    image : numpy.ndarray
        A image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    Returns
    -------
    NONE

    """
    global OUT_FOLDER
    OUT_FOLDER = directory
    
    # set height/width of image
    height, width = image.shape[:2]

    # step 1: smooth the image
    smooth_img = median_filter(image)

    # step 2: convert to grayscale for edge detection
    gray_img = cv_gray(smooth_img)
    gray_img2 = cv_gray(image)

    # step 3: detect edges
    edge = canny_edge(gray_img) # leverages opencv canny
    edge2 = custom_edge_gaussian(gray_img2)
    grad, direction = gradient_intesity(gray_img) # custom edge canny
    suppressed = non_max_suppression(grad, direction)
    threshold, weak, strong = double_threshold(suppressed, 100, 20)
    edge3 = edge_track(threshold, weak)

    # step 4: dilate edges
    dilated_edge = edge_dilation(edge3)

    # step 5: threshold edges
    thresh_edge = threshold_func(dilated_edge)

    # step 6 downsample smoothed BGR image
    downsampled = downsample(smooth_img)

    # step 7: bilateral filter downsampled image
    filtered = bi_filter(downsampled)

    # step 8: upsample filter image
    upsampled = upsample(filtered, height, width)

    # step 9: median filter
    smooth_up = median_filter_upsample(upsampled)

    # step 10: quantize colors
    color_adj = quantize_colors(smooth_up, height, width)

    # step 11: combine
    combine(color_adj, thresh_edge, height, width)