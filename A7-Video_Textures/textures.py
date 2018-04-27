""" Video Textures

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
import scipy.signal


def videoVolume(images):
    """ Create a video volume (4-d numpy array) from the image list.

    Parameters
    ----------
    images : list
        A list of frames. Each element of the list contains a numpy array
        representing a color image. You may assume that each frame has the same
        shape: (rows, cols, 3).

    Returns
    -------
    numpy.ndarray(dtype: np.uint8)
        A 4D numpy array. This array should have dimensions
        (num_frames, rows, cols, 3).
    """

    rows, cols = images[0].shape[:2]
    num_images = len(images)

    # create new 4-d array for output
    volume = np.zeros((num_images, rows, cols, 3), dtype=np.uint8)

    # set the videovolume return values
    for i in xrange(num_images):
        volume[i] = images[i]

    return volume


def computeSimilarityMetric(video_volume):
    """Compute the differences between each pair of frames in the video volume.

    The goal, of course, is to be able to tell how good a jump between any two
    frames might be so that the code you write later on can find the optimal
    loop. The closer the similarity metric is to zero, the more alike the two
    frames are.

    Loop through each pair (i, j) of start and end frames in the video volume.
    Calculate the root sum square deviation (rssd) score for each pair and
    store the value in cell (i, j) of the output:

        rssd = sum( (start_frame - end_frame) ** 2 ) ** 0.5

    Finally, divide the entire output matrix by the average value of the matrix
    in order to control for resolution differences and distribute the values
    over a consistent range.

    Hint: Remember the matrix is symmetrical, so when you are computing the
    similarity at i, j, its the same as computing the similarity at j, i so
    you don't have to do the math twice.  Also, the similarity at all i,i is
    always zero, no need to calculate it.

    Parameters
    ----------
    video_volume : numpy.ndarray
        A 4D numpy array with dimensions (num_frames, rows, cols, 3).

        This can be produced by the videoVolume function.

    Returns
    -------
    numpy.ndarray(dtype: np.float64)
        A square 2d numpy array where output[i,j] contains the similarity
        score between the start frame at i and the end frame at j of the
        video_volume.  This matrix is symmetrical with a diagonal of zeros.
    """

    # set the num of images and empty arrays for editing
    num_images = video_volume.shape[0]
    ssd_array = np.zeros((num_images, num_images), dtype=np.float64)
    adj_vv = video_volume.astype(np.float64)

    # iterate throught the images and compute the rssd value for each pair
    for i in xrange(num_images):

        # set the first frame per the equation above
        start_frame = adj_vv[i]
        for j in xrange(num_images):

            # set the last frame and calculate and store the rssd values for each pixel
            end_frame = adj_vv[j]
            ssd_array[i][j] = np.sum((start_frame - end_frame) ** 2) ** 0.5

    # normalize by the average across the entire matrix
    average = ssd_array.mean() # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matrix.mean.html
    ssd_array = ssd_array/average

    return ssd_array


def transitionDifference(similarity):
    """Compute the transition costs between frames accounting for dynamics.

    Iterate through each cell (i, j) of the similarity matrix (skipping the
    first two and last two rows and columns).  For each cell, calculate the
    weighted sum:

        diff = sum ( binomial * similarity[i + k, j + k]) for k = -2...2

    Hint: There is an efficient way to do this with 2d convolution. Think about
          the coordinates you are using as you consider the preceding and
          following frame pairings.

    Parameters
    ----------
    similarity : numpy.ndarray
        A similarity matrix as produced by your similarity metric function.

    Returns
    -------
    numpy.ndarray
        A difference matrix that takes preceding and following frames into
        account. The output difference matrix should have the same dtype as
        the input, but be 4 rows and columns smaller, corresponding to only
        the frames that have valid dynamics.
    """

    # set the rows and cols to calculate the new size
    rows, cols = similarity.shape[:2]

    # instantiate a numpy array 4 smaller in rows and cols as noted above
    transition_costs = np.zeros((rows-4, cols-4), dtype=similarity.dtype)

    # for the convolution kernel to work it needs to be in a matrix
    # as the binomial function returns an array, by grabbing the diagonal
    # and convolving it will effectively multiple by the similarity as noted
    # above
    kernel = np.diag(binomialFilter5()) # https://docs.scipy.org/doc/numpy/reference/generated/numpy.diag.html

    # leverage the convlution2d method as hinted above with the diagonal kernel output
    # https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.signal.convolve2d.html
    transition_costs = scipy.signal.convolve2d(similarity, kernel, 'valid')

    return transition_costs



def findBiggestLoop(transition_diff, alpha):
    """Find the longest and smoothest loop for a given the difference matrix.

    For each cell (i, j) in the transition differences matrix, find the
    maximum score according to the following metric:

        score = alpha * (j - i) - transition_diff[j, i]

    The pair i, j correspond to the start and end indices of the longest loop.

    **************************************************************************
      NOTE: Remember to correct the indices from the transition difference
        matrix to account for the rows and columns dropped from the edges
                    when the binomial filter was applied.
    **************************************************************************

    Parameters
    ----------
    transition_diff : np.ndarray
        A square 2d numpy array where each cell contains the cost of
        transitioning from frame i to frame j in the input video as returned
        by the transitionDifference function.

    alpha : float
        A parameter for how heavily you should weigh the size of the loop
        relative to the transition cost of the loop. Larger alphas favor
        longer loops, but may have rough transitions. Smaller alphas give
        shorter loops, down to no loop at all in the limit.

    Returns
    -------
    int, int
        The pair of (start, end) indices of the longest loop after correcting
        for the rows and columns lost due to the binomial filter.
    """

    # set the rows and cols for iteration
    rows, cols = transition_diff.shape[:2]

    # set indexes to store the current best score indexes
    start, end = 0, 0

    # begin with the default score set at the origin point
    old_score = alpha * (0 - 0) - transition_diff[0, 0]

    # loop through the transition matrix and calculate the best score
    for i in xrange(rows):
        for j in xrange(cols):

            # calculate the score at the current position as the new_score
            new_score = alpha * (j - i) - transition_diff[j, i]

            # if the current position's score is higher set it
            if new_score > old_score:

                # set the indexes that will be returned to the new best position
                start, end = i, j

                # set the new best score and iterate
                old_score = new_score

    # pad the indexes due to the removal of rows/columns as part of convolution
    start = start + 2
    end = end + 2

    # print("alpha: %s start: %s, end: %s", alpha, start, end)
    return start, end


def synthesizeLoop(video_volume, start, end):
    """Pull out the given loop from the input video volume.

    Parameters
    ----------
    video_volume : np.ndarray
        A (time, height, width, 3) array, as created by your videoVolume
        function.

    start : int
        The index of the starting frame.

    end : int
        The index of the ending frame.

    Returns
    -------
    list
        A list of arrays of size (height, width, 3) and dtype np.uint8,
        similar to the original input to the videoVolume function.
    """
    # set an append list to return
    loop = []

    # iterate through the start and end ranges and append the frames and return
    for i in xrange(start, end+1):
        loop.append(video_volume[i])
    return loop


def binomialFilter5():
    """Return a binomial filter of length 5.

    NOTE: DO NOT MODIFY THIS FUNCTION.

    Returns
    -------
    numpy.ndarray(dtype: np.float)
        A 5x1 numpy array representing a binomial filter.
    """
    return np.array([1 / 16., 1 / 4., 3 / 8., 1 / 4., 1 / 16.], dtype=float)
