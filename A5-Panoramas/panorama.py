""" Panoramas

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but the functions should NOT save the image to file.

    2. DO NOT import any other libraries aside from those that we provide.
    You may not import anything else, and you should be able to complete
    the assignment with the given libraries (and in many cases without them).

    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.

    4. This file has only been tested in the provided virtual environment.
    You are responsible for ensuring that your code executes properly in the
    virtual machine environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""
import numpy as np
import scipy as sp
import cv2


def getImageCorners(image):
    """Return the x, y coordinates for the four corners bounding the input
    image and return them in the shape expected by the cv2.perspectiveTransform
    function. (The boundaries should completely encompass the image.)

    Parameters
    ----------
    image : numpy.ndarray
        Input can be a grayscale or color image

    Returns
    -------
    numpy.ndarray(dtype=np.float32)
        Array of shape (4, 1, 2).  The precision of the output is required
        for compatibility with the cv2.warpPerspective function.

    Notes
    -----
        (1) Review the documentation for cv2.perspectiveTransform (which will
        be used on the output of this function) to see the reason for the
        unintuitive shape of the output array.

        (2) When storing your corners, they must be in (X, Y) order -- keep
        this in mind and make SURE you get it right.
    """

    # instantiate empty corners np array, and set the shape variables of the input image
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    (height, width) = image.shape[:2]

    # iterate through the four corners and set them into the output np array
    corners[0,0,:] = [0, 0] # top left corner (origin)
    corners[1,0,:] = [0, height] # bottom left corner
    corners[2,0,:] = [width, height] # top right corner
    corners[3,0,:] = [width, 0] # bottom right corner

    # return the corners
    return corners


def findMatchesBetweenImages(image_1, image_2, num_matches):
    """Return the top list of matches between two input images.

    Parameters
    ----------
    image_1 : numpy.ndarray
        The first image (can be a grayscale or color image)

    image_2 : numpy.ndarray
        The second image (can be a grayscale or color image)

    num_matches : int
        The number of keypoint matches to find. If there are not enough,
        return as many matches as you can.

    Returns
    -------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_1

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors from image_2

    matches : list<cv2.DMatch>
        A list of the top num_matches matches between the keypoint descriptor
        lists from image_1 and image_2

    Notes
    -----
        (1) You will not be graded for this function.
    """
    feat_detector = cv2.ORB(nfeatures=500)
    image_1_kp, image_1_desc = feat_detector.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = feat_detector.detectAndCompute(image_2, None)
    bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bfm.match(image_1_desc, image_2_desc),
                     key=lambda x: x.distance)[:num_matches]
    return image_1_kp, image_2_kp, matches


def findHomography(image_1_kp, image_2_kp, matches):
    """Returns the homography describing the transformation between the
    keypoints of image 1 and image 2.

        ************************************************************
          Before you start this function, read the documentation
                  for cv2.DMatch, and cv2.findHomography
        ************************************************************

    Follow these steps:

        1. Iterate through matches and store the coordinates for each
           matching keypoint in the corresponding array (e.g., the
           location of keypoints from image_1_kp should be stored in
           image_1_points).

            NOTE: Image 1 is your "query" image, and image 2 is your
                  "train" image. Therefore, you index into image_1_kp
                  using `match.queryIdx`, and index into image_2_kp
                  using `match.trainIdx`.

        2. Call cv2.findHomography() and pass in image_1_points and
           image_2_points, using method=cv2.RANSAC and
           ransacReprojThreshold=5.0.

        3. cv2.findHomography() returns two values: the homography and
           a mask. Ignore the mask and return the homography.

    Parameters
    ----------
    image_1_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the first image

    image_2_kp : list<cv2.KeyPoint>
        A list of keypoint descriptors in the second image

    matches : list<cv2.DMatch>
        A list of matches between the keypoint descriptor lists

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between image_1 and image_2
    """

    # instantiate empty np arrays for the image points for matching keypoints   
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    
    # iterate through the input keypoint arrays and grab the matching pairs
    for i in xrange(len(matches)):
        image_1_points[i] = image_1_kp[matches[i].queryIdx].pt # https://stackoverflow.com/questions/35884409/how-to-extract-x-y-coordinates-from-opencv-cv2-keypoint-object
        image_2_points[i] = image_2_kp[matches[i].trainIdx].pt
    
    # call the homography function per the notes above and set the output array and mask (which is not used)
    hom_array, unused_mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=5.0)    
    return hom_array


def getBoundingCorners(corners_1, corners_2, homography):
    """Find the coordinates of the top left corner and bottom right corner of a
    rectangle bounding a canvas large enough to fit both the warped image_1 and
    image_2.

    Given the 8 corner points (the transformed corners of image 1 and the
    corners of image 2), we want to find the bounding rectangle that
    completely contains both images.

    Follow these steps:

        1. Use the homography to transform the perspective of the corners from
           image 1 (but NOT image 2) to get the location of the warped
           image corners.

        2. Get the boundaries in each dimension of the enclosing rectangle by
           finding the minimum x, maximum x, minimum y, and maximum y.

    Parameters
    ----------
    corners_1 : numpy.ndarray of shape (4, 1, 2)
        Output from getImageCorners function for image 1

    corners_2 : numpy.ndarray of shape (4, 1, 2)
        Output from getImageCorners function for image 2

    homography : numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between image_1 and image_2

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        2-element array containing (x_min, y_min) -- the coordinates of the
        top left corner of the bounding rectangle of a canvas large enough to
        fit both images (leave them as floats)

    numpy.ndarray(dtype=np.float64)
        2-element array containing (x_max, y_max) -- the coordinates of the
        bottom right corner of the bounding rectangle of a canvas large enough
        to fit both images (leave them as floats)

    Notes
    -----
        (1) The inputs may be either color or grayscale, but they will never
        be mixed; both images will either be color, or both will be grayscale.

        (2) Python functions can return multiple values by listing them
        separated by commas.

        Ex.
            def foo():
                return [], [], []
    """
    # set the min and max value return np arrays
    min_vals = np.zeros((2), dtype=np.float64)
    max_vals = np.zeros((2), dtype=np.float64)

    # set the adjusted corners from perspectiveTransform
    adj_corners = cv2.perspectiveTransform(corners_1, homography)
    
    # calculate the min and max x and y values for the adjusted corners
    min_x_adj = min(adj_corners[:,0,0]) # https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html 
    max_x_adj = max(adj_corners[:,0,0])
    min_y_adj = min(adj_corners[:,0,1])
    max_y_adj = max(adj_corners[:,0,1])
    
    # calculate the min and max x and y values for the second image
    min_cor2x = min(corners_2[:,0,0])
    max_cor2x = max(corners_2[:,0,0])
    min_cor2y = min(corners_2[:,0,1])
    max_cor2y = max(corners_2[:,0,1])

    # determine the overall min and max coordinates and set them into the output arrays
    min_vals[0] = min(min_x_adj, min_cor2x)
    min_vals[1] = min(min_y_adj, min_cor2y)
    max_vals[0] = max(max_x_adj, max_cor2x)
    max_vals[1] = max(max_y_adj, max_cor2y)

    return min_vals, max_vals

def warpCanvas(image, homography, min_xy, max_xy):
    """Warps the input image according to the homography transform and embeds
    the result into a canvas large enough to fit the next adjacent image
    prior to blending/stitching.

    Follow these steps:

        1. Create a translation matrix (numpy.ndarray) that will shift
           the image by x_min and y_min. This looks like this:

            [[1, 0, -x_min],
             [0, 1, -y_min],
             [0, 0, 1]]

        2. Compute the dot product of your translation matrix and the
           homography in order to obtain the homography matrix with a
           translation.

        NOTE: Matrix multiplication (dot product) is not the same thing
              as the * operator (which performs element-wise multiplication).
              See Numpy documentation for details.

        3. Call cv2.warpPerspective() and pass in image 1, the combined
           translation/homography transform matrix, and a vector describing
           the dimensions of a canvas that will fit both images.

        NOTE: cv2.warpPerspective() is touchy about the type of the output
              shape argument, which should be an integer.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale or color image (test cases only use uint8 channels)

    homography : numpy.ndarray(dtype=np.float64)
        A 3x3 array defining a homography transform between two sequential
        images in a panorama sequence

    min_xy : numpy.ndarray(dtype=np.float64)
        2x1 array containing the coordinates of the top left corner of a
        canvas large enough to fit the warped input image and the next
        image in a panorama sequence

    max_xy : numpy.ndarray(dtype=np.float64)
        2x1 array containing the coordinates of the bottom right corner of
        a canvas large enough to fit the warped input image and the next
        image in a panorama sequence

    Returns
    -------
    numpy.ndarray(dtype=image.dtype)
        An array containing the warped input image embedded in a canvas
        large enough to join with the next image in the panorama; the output
        type should match the input type (following the convention of
        cv2.warpPerspective)

    Notes
    -----
        (1) You must explain the reason for multiplying x_min and y_min
        by negative 1 in your writeup.
    """
    # canvas_size properly encodes the size parameter for cv2.warpPerspective,
    # which requires a tuple of ints to specify size, or else it may throw
    # a warning/error, or fail silently
    canvas_size = tuple(np.round(max_xy - min_xy).astype(np.int))

    # set the translation matrix as noted in the above comments
    translation = np.array([[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1]])
   
    # compute the dot product of the homography and translation matrices
    hot = np.dot(translation, homography) # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html 

    # compute the warped image and return it
    warped = cv2.warpPerspective(image, hot, canvas_size) # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html 
    
    return warped


def blendImagePair(image_1, image_2, num_matches):
    """This function takes two images as input and fits them onto a single
    canvas by performing a homography warp on image_1 so that the keypoints
    in image_1 aligns with the matched keypoints in image_2.

    **************************************************************************

        You MUST replace the basic insertion blend provided here to earn
                         credit for this function.

       The most common implementation is to use alpha blending to take the
       average between the images for the pixels that overlap, but you are
                    encouraged to use other approaches.

           Be creative -- good blending is the primary way to earn
                  Above & Beyond credit on this assignment.

    **************************************************************************

    Parameters
    ----------
    image_1 : numpy.ndarray
        A grayscale or color image

    image_2 : numpy.ndarray
        A grayscale or color image

    num_matches : int
        The number of keypoint matches to find between the input images

    Returns:
    ----------
    numpy.ndarray
        An array containing both input images on a single canvas

    Notes
    -----
        (1) This function is not graded by the autograder. It will be scored
        manually by the TAs.

        (2) The inputs may be either color or grayscale, but they will never be
        mixed; both images will either be color, or both will be grayscale.

        (3) You can modify this function however you see fit -- e.g., change
        input parameters, return values, etc. -- to develop your blending
        process.
    """

    # calculate the keypoints and identified matches
    kp1, kp2, matches = findMatchesBetweenImages(
        image_1, image_2, num_matches)
    
    # calculate the homography matrix using the keypoint and match arrays
    homography = findHomography(kp1, kp2, matches)

    # get the corners of the two images
    corners_1 = getImageCorners(image_1)
    corners_2 = getImageCorners(image_2)

    # adjust the bounds of the combined images based on the coordinates and the homography offset
    min_xy, max_xy = getBoundingCorners(corners_1, corners_2, homography)

    # compute the warped image and set it to the output
    output_image = warpCanvas(image_1, homography, min_xy, max_xy)
    # set iterators to ints and set the height/width of the second image
    min_xy = min_xy.astype(np.int)
    (height, width) = image_2.shape[:2]

    
    # instantiate a new warped_temp and final blend array for iteration for blending
    warped_temp = output_image[-min_xy[1]:,-min_xy[0]:] # set the warped_temp to the output_image
    warped_temp = warped_temp[:height, :width] #instantiate it to origin point with full canvas height and width

    # create the alpha_blend holding array
    alpha_blend = image_2

    # iterate through the rows and columns of the images relative to the second image
    for i in xrange(height):
        for j in xrange(width):
            # if the BGR pixel values at the position are not 0, i.e. black, then proceed to blend
            if warped_temp[i][j].any() != 0:
                # take the average value of the pixels of the two images at the point and set it in the final output for basic alpha blend
                alpha_blend[i][j] = np.mean(np.array([warped_temp[i][j], image_2[i][j]]), axis=0)  # https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
                                                                                                     # https://stackoverflow.com/questions/17079279/how-is-axis-indexed-in-numpys-array

    # stitch the two images together in the overall canvas
    output_image[-min_xy[1]:-min_xy[1] + height,
                -min_xy[0]:-min_xy[0] + width] = alpha_blend

    return output_image
    # END OF FUNCTION

# DOES NOT USE
# https://docs.opencv.org/3.2.0/d0/d86/tutorial_py_image_arithmetics.html
def Mask_Calc(image_2, warp):
    """This function was a second attempt at blending leveraging the bitwise not operations and a generated mask.
    Didn't really give as good of an output as the basic mean/alpha blend. I didn't end up using it, included here
    for reference and I added the resulting output into the presentation.
"""
    # Now create a mask of of image_2 and create its inverse
    img2gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # create roi to position image_2
    roi = cv2.bitwise_and(image_2, image_2, mask=mask)

    # blackout area of warp
    im2 = cv2.bitwise_and(warp, warp, mask=mask_inv)

    # add the images back together
    result = cv2.add(im2, roi)

    return result

