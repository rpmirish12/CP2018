import cv2

'''This snippet of code imports a set of nine images and then applies a basic alpha blend to them 
   creating an output image that is the lightened combination of a single light source that was
   shifted around each of the images.

   Author: Ryan Miller
   Purpose: GT Computational Photography
   Date: 1/22/2018'''


if __name__ == '__main__' :
    #Load in the image paths to store as an python list
    img_locations = ['raw_images/IMG_0779.jpg','raw_images/IMG_0780.jpg','raw_images/IMG_0781.jpg','raw_images/IMG_0782.jpg','raw_images/IMG_0783.jpg','raw_images/IMG_0784.jpg','raw_images/IMG_0785.jpg','raw_images/IMG_0786.jpg','raw_images/IMG_0787.jpg']
    img_array = []

    #Append the images to the list
    for location in img_locations:
        im = cv2.imread(location).astype(float)
        img_array.append(im)
    
    #Apply an alpha blend addition lowering the light intesity by 80% for each of the 9 images in the set
    Outimage = img_array[0]*0.2 + img_array[8]*0.2 + img_array[1]*0.2 + img_array[2]*0.2 + img_array[3]*0.2 + img_array[4]*0.2 + img_array[5]*0.2 + img_array[6]*0.2 + img_array[7]*0.2 + img_array[8]*0.2    
    
    #write the resultant image to a file
    cv2.imwrite("alpha_python_image.jpg", Outimage)