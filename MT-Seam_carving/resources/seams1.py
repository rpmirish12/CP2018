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


import numpy as np
import cv2
import sys
import io
from datetime import datetime


def energy(image, image_name):
    """Function that takes in an BGR OpenCV image and calculates the sobel 
    derivative values. 

    This output matrix's values contains the intensity of change different
    between 2 neighboring pixels. The higher the change, the higher the
    value stored in the matrix.

    INPUT: <cv2.image> <string>
    OUTPUT: <cv2.image>

    """

    # convert image to grayscale for performance
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY) 

    # calculate the energy values in the x direction
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 3) 
    abs_x = cv2.convertScaleAbs(dx)

    # calculate the energy values in the y direction
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 3) 
    abs_y = cv2.convertScaleAbs(dy)

    # weight and add dx,dy abs_values to generate output matrix
    output = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0) 
    
    return output


def color_map(image, image_name):
    """NOT COMPLETED!
    
    This function would generate a t_map for the outputted energy calculation
    of both the seam values in the horizontal and vertical direction.

    INPUT: <cv2.image> <string>
    OUTPUT: <np.array>

    """

    # convert to heatmap
    image_out = cv2.applyColorMap(image, cv2.COLORMAP_JET)

    # write to disk
    t_name = image_name + "_tmap.png"
    cv2.imwrite(t_name, image_out.astype(np.uint8))


def remove_seam(image, image_name, pixels):
    """Main looping function of program that drives logic for seam 
    identification, seam storage, seam removal, and seam addition.

    It natively returns a new image that has seams removed from it.

    INPUT: <cv2.image>, <string>, <int>
    OUTPUT: <cv2.image>

    """

    new_img = image # load in reference to new_image
    removed_seams = [] # set a list to store removed seams

    # calculate and store the energy matrix for seam analysis, write to disk
    image_energy = energy(new_img, image_name) 
    out_energy_name = image_name + "_energy.png"
    cv2.imwrite(out_energy_name, image_energy.astype(np.uint8))

    # calculate and store the weighted value map matrix of the image
    value_map = vertical_map(image_energy)

    # NOT USED: would create t_map here
    color_map(image_energy, image_name)

    # Iterate through user entered pixel amount, create seams, store them, and
    # generate new_img output with seams removed
    for pixel in xrange(pixels):

        # calculate weighted energy matrix
        image_energy = energy(new_img, image_name)
        value_map = vertical_map(image_energy)

        # calculate and store seams with their total cost
        # faster implentations written here: (seam, cost) = fast_calc_seam(value_map)
        (list_seams, cost_seams) = calc_seams(value_map)
        (new_img, removed_seam) = seam_removal(new_img, list_seams, cost_seams)

        # store seams that were removed in a list for use in drawing them to 
        # check accuracy, and in add seams to generate a larger image
        removed_seams.append(removed_seam)

    # adjust the seam values by offset and draw them onto the image    
    adj_removed_seams = adj_seams(image, removed_seams, image_name)

    # expand the image adding removed seams into original
    add_seams(image, adj_removed_seams, pixels, image_name)

    # return the smaller, seam_carved image
    return new_img


def vertical_map(energy):
    """Function that takes in an energy matrix, and starting from the top,
     iterates through the matrix down calculating, storing, and returning a 
     new matrix that contains weighted energy values for seam identification

    INPUT: <np.array[][]>
    OUTPUT: <np.array[][]>

    """

    # Determine the size of the image at start and calculate the seams
    energy = np.int_(energy)
    (rows, columns) = energy.shape[:2]

    # Generate a numpy matrix of zeroes that is the same size as the input
    # energy matrix
    vert_map = np.zeros((rows, columns))

    # Set the first entire row equal for both matrices following dynamic 
    # programming visual: https://en.wikipedia.org/wiki/Seam_carving
    for col in xrange(columns):
        vert_map[0][col] = energy[0][col]
    
    # Iterate through the matrix selecting the min 8-pixel up and adding it's
    # value to the original, proceeding through each column in the row
    for row in xrange(1, rows):
        for col in xrange(columns):
            if(col == 0):
                vert_map[row][col] = energy[row][col] + min(energy[row-1][col], energy[row-1][col+1])
            if (col == columns-1):
                vert_map[row][col] = energy[row][col] + min(energy[row-1][col], energy[row-1][col-1])
            else:
                vert_map[row][col] = energy[row][col] + min(energy[row-1][col-1], energy[row-1][col], energy[row-1][col+1])
    
    # return the generated, weighted map               
    return vert_map


def calc_seams(value_map):
    """Function that takes in an weighted energy matrix, and starting from the 
    bottom, iterates through the matrix.
     
    Calculating, storing, and returning a list of all the possible seams from
    bottom row to top row in the image, and the associated cost of each seam.

    INPUT: <np.array[][]>
    OUTPUT: <list(np.array[])>, <list(int)>

    """
    # determine the size of the input matrix for iteration 
    (rows, columns) = value_map.shape[:2]

    # convert to int for row/column indexing
    value_map = np.int_(value_map)

    # instantiate the list of seams and the associate costs for return
    list_costs = []
    list_seams = []

    # loop through each column in the matrix starting at the 0 posion
    for col in xrange(columns):

        # create a 1-D array of zeros as the seam np.array
        seam = np.zeros(rows)

        # set the current active column for use in row iteration
        active_column = col
        
        # reset the seam cost to 0 and the starting seam position to the last 
        # col index in the seam for each iteration
        seam_cost = 0
        seam[-1] = active_column

        # set the intial cost for the value initially loaded
        seam_cost+=value_map[rows-1][active_column]

        # iterate over each row starting at the second to last row to the top
        for row in xrange(rows-2, -1, -1):

            # check if the next value is in the 0 column
            if (seam[row+1] == 0):

                # adjust to compare to just the two position available: above, above and right
                min_vals = (value_map[row][active_column], value_map[row][active_column+1])

                # store the index of the least energy value of the 2 above
                seam[row] = np.argmin(min_vals)

                # set the active column for next iteration and storage
                active_column = np.argmin(min_vals)
            
            # check if next value is in the last column
            elif (active_column == columns-1):

                # adjust to compare to just the two position available: above, above and left
                min_vals = (value_map[row][active_column], value_map[row][active_column-1])

                # index position is relative to the comparison, to get the actual index
                # convert relative to the last position based on which value is the min
                if(np.argmin(min_vals)==0):
                    active_column = columns-1
                else:
                    active_column = columns-2
                
                # set the active column for next iteration and storage
                seam[row] = active_column

            # main operation for all columns other than first or last
            else:

                #compare to all three available positions: above and right, above, above and left
                min_vals = (value_map[row][active_column-1], value_map[row][active_column], value_map[row][active_column+1])
                
                # index position is relative to the comparison, to get the actual index
                # convert relative to the prior active_column position based on which 
                # value is the min
                if (np.argmin(min_vals)==0):
                    active_column -= 1
                elif (np.argmin(min_vals)==1):
                    active_column = active_column
                else:
                    active_column += 1

                # set the active column for next iteration and storage
                seam[row] = active_column

            # based on selected value add selected path cost to container
            seam_cost += value_map[row][active_column]

        # append the total seam energy cost and the seam itself to respecive
        # lists and return once all seams have been computed    
        list_costs.append(seam_cost)
        list_seams.append(seam)
    return (list_seams, list_costs)


def adj_seams(image, list_seams, image_name):
    """Function that adjusts the seam position based on offset and also
    outputs and image that has the seams drawn in red.

    INPUT: <np.array[][]>, <list(np.array)>, <string>
    OUTPUT: <list(np.array[])>

    """

    # determine the size of the input matrix for iteration 
    (height, width) = image.shape[:2]

    # set to integer values for iteration indexing
    image = np.int64(image)

    # create an image copy to draw lines into image
    tmp_image = image.copy()

    # set to integer values for iteration indexing
    list_seams = np.int_(list_seams)

    # iterate over each seam in the image
    for i in xrange(len(list_seams)):

        # grab an individual seam
        seam = list_seams[i]

        # iterate over the rows in the seam
        for row in xrange(height):

            # grab the column index stored at the row, adjust if over limit
            col = seam[row]
            if(col >= width-1):
                col = width-1
            
            # draw the seam onto the image by shifting bgr value to red
            tmp_image[row][col] = (0,0,255)

            # adjust all seams in list to the right of the seam value added
            for j in xrange(len(list_seams)):
                if(list_seams[j][row] >= col):
                    list_seams[j][row] = list_seams[j][row]+1
    
    # write image with seams overlayed to file and return adjusted list of seams
    seam_name = image_name + "_seams.png"
    cv2.imwrite(seam_name, tmp_image.astype(np.uint8))
    return list_seams

def add_seams(image, adj_seams, pixels, image_name):
    """Function that expands the image by leveraging the seam removal
    matrixes and iterating through the original image.

    At each position this function checks to see where to insert, adjusts
    the offset and insterts the seam in the order they were generated in
    the removal step. 

    Finally the function writes the expanded image to disk.

    INPUT: <np.array[][]>, <np.array[][]>, <int>, <string>
    OUTPUT: NONE
    """
    # determine the size of the input matrix for iteration
    (height, width) = image.shape[:2]
    add_image=image

    # convert seams to integers for indexing
    list_seams = np.int_(adj_seams)

    # iterate through each column of pixel to add
    for i in xrange(pixels):

        # grab the relevant seam from the list
        seam = list_seams[i]

        # create a new matrix one column wider for seam insertion
        zero_image=np.zeros((height,width+1,3))

        # iterate over each row and calculate place to add seam
        for row in xrange(height):

            # grab the column index for seam insertion
            column = seam[row]

            # if at the 0 column adjust based on average of left and right pixels
            # and insert after 0th value leveraging array slicing
            # https://stackoverflow.com/questions/509211/understanding-pythons-slice-notation
            if (column == 0):

                # grab the average bgr tuple of the surroudning pixels
                avg_bgr = (add_image[row, column, :3]+add_image[row, column+1, :3])/2

                # to the left 0 value set to the orignal image value
                zero_image[row, column, :3] = add_image[row, column, :3]

                # position to insert the seam value average
                zero_image[row, column+1, :3] = avg_bgr

                # for all other positions
                zero_image[row, column+1:, :3] = add_image[row, column:]
            else:
                # ran into and index bug due to iteration, if seam position goes
                # out of bounds just add to the end, THIS IS NOT A PERMANENT FIX
                if(column >= height):
                    column=height-1
                
                # grab the average bgr tuple of the surroudning pixels
                avg_bgr = (add_image[row, column-1, :3]+add_image[row, column, :3])/2

                # to the left of the insertion, grab bgr values from original image
                zero_image[row, :column, :3] = add_image[row, :column, :3]

                # position to insert the seam value average
                zero_image[row, column, :3] = avg_bgr

                # for all other positions
                zero_image[row,column+1:, :3] = add_image[row, column:, :3]

        # set the image to it's new expanded size, and iterate one pixel column
        # at a time until total amount is acheived
        add_image=np.copy(zero_image)
        width=width+1
    
    # write the expanded output image with seams inserted to disk
    expand_name = image_name + "_expanded.png"
    cv2.imwrite(expand_name, add_image.astype(np.uint8))



def seam_removal(image, list_seams, cost_seams):
    """Function that removes the lowest energy value seam from the image
    and returns the image and the seam removed for later use.

    INPUT: <np.array[][]>, <np.array[][]>, <int>
    OUTPUT: <np.array[][]>, <np.array()>
    """
    # determine the size of the input matrix for iteration
    (height, width) = image.shape[:2]

    # grab the lowest valued seam in the list by getting the lowest
    # cost index with argsort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
    sort_index = np.argsort(cost_seams)
    column = sort_index[0]

    # convert to int for indexing and grab the seam
    column = np.int_(column)
    seam = list_seams[column]

    # create a new np.array one column smaller to store the new image
    newimg = np.zeros((height, width-1, 3))

    # indexes for deletion of the individual channel elements
    b,g,r = 0,1,2

    # iterate over the rows leveraging the column index to remove
    # the seam pixels at each position.
    for col in xrange(height):
        column = seam[col]

        # calls np.delete on each channel, returning all other available values
        # and storing into each row in the image with some nice array slicing!
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.delete.html
        newimg[col, :, b] = np.delete(image[col, :, b], column)
        newimg[col, :, g] = np.delete(image[col, :, g], column)
        newimg[col, :, r] = np.delete(image[col, :, r], column)     

    # return the image minus the seam, and the seam removed   
    return (newimg, seam)

def seam_carve(image, image_name, pixels):
    """Driver function that kicks off the program.

    INPUT: <np.array[][]>, <string>, <int>
    OUTPUT: <np.array[][]>
    """
    startTime = datetime.now()
    seam_removal_img = remove_seam(image, image_name, pixels)
    print datetime.now() - startTime 
    return seam_removal_img

""" BELOW IS FASTER IMPLEMENTATION OF ABOVE FUNCTIONS
    
    Not selected as final due to the output not being as clean.
    This is due to the only main difference being less iterations as it
    doesn't rely on the summed energy cost for each seam.
"""

"""
def fast_calc_seam(value_map):
    # Generate a list of seams with energy values
    (rows, columns) = value_map.shape[:2]
    value_map = np.int_(value_map)

    seam = np.zeros(rows)
    active_column = np.argmin(value_map[-1])
    seam_cost = 0
    seam[-1] = active_column
    # print(seam)
    seam_cost+=value_map[rows-1][active_column]
    for row in xrange(rows-2, -1, -1):
        if (seam[row+1] == 0):
            min_vals = (value_map[row][active_column], value_map[row][active_column+1])
            seam[row] = np.argmin(min_vals)
            active_column = np.argmin(min_vals)
                # print("I'm in the 0 column", min_vals, "active columns", np.argmin(min_vals))
        elif (active_column == columns-1):
            min_vals = (value_map[row][active_column], value_map[row][active_column-1])
            if(np.argmin(min_vals)==0):
                active_column = columns-1
            else:
                active_column = columns-2
            seam[row] = active_column
                # print("I'm in the last column", min_vals, "active column:", active_column)
        else:
            min_vals = (value_map[row][active_column-1], value_map[row][active_column], value_map[row][active_column+1])
            if (np.argmin(min_vals)==0):
                active_column -= 1
            elif (np.argmin(min_vals)==1):
                active_column = active_column
            else:
                active_column += 1
            seam[row] = active_column
        seam_cost += value_map[row][active_column]
    return (seam, seam_cost)

"""
