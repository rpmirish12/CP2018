# Just a Photograph

## Synopsis

In this assignment you will submit one of your own photographs that to demonstrate the workflow for assignments using the course VM and other tools. Your picture may have been taken anywhere and any time (ie. it does not have to be in the last few days). However, you must have access to the camera settings (e.g., EXIF data) used for the photo, and *you* must be the photographer.

You may choose your personal best photograph, or a photograph that you wish you had been able to do something better with. (“Better” here could be defined as using a "computational photography" process, or "better" aesthetically, or other idea reasonable definition.) Make sure that the photograph you share is (a) yours and (b) something you are comfortable sharing. (Please, nothing controversial and/or offensive - use your best judgement.)


### Cameras and EXIF Data

You are allowed to use any kind of camera  -- anything from smartphones to high-end DSLR cameras -- for this class. Some assignments require granular control of aperture, shutter speed, ISO, and other settings that can be difficult to adjust for smartphones; other assignments require very stable positioning of the camera between successive shots that can be challenging without a tripod. However, there is no required hardware for this course -- many students successfully complete all of the assignments using only a smartphone.

To help you get used to your camera and its settings, you are asked to provide some technical information about your photograph.  You should immediately find out how to get the EXIF data for the camera you plan to use in this course. You will need to be able to access:

  - Exposure time (ex: 1/2000 s, 1/30 s)
  - Aperture (ex: f 2.8, f16) 
  - ISO (ex: ISO 100, ISO 1600)

EXIF data is recorded in digital images, and can generally be easily found on your phone, in your digital camera, or on your computer. Image editing can affect the data, so record the settings before playing with your image.  Search online for information on finding EXIF data for your device if you are not sure how to find it or don’t know what it is. You can also discuss EXIF data and where to find it on Piazza.


## Instructions

### 1. Choose a Photograph

Select a photograph you have taken to submit for this assignment and put it in a zip file named `resources.zip` in the project directory.


### 2. Complete the Report

Make a copy of the [report template](https://drive.google.com/open?id=13c8wkqLOQ6EqfQd2bJOSVwnuPD9sPOrnKl8dGwJJ9rg) and answer all of the questions. Save your report as `report.pdf` in the project directory. 


### 3. Submit the Project

Make sure that files named `report.pdf` and `resources.zip` containing your image are in the assignment folder -- both files will be uploaded. (These names are important because the submission script searches the current working directory for exact matches to the filenames). Your image must be one of the following types: jpg, jpeg, bmp, png, tif, or tiff.

**Note:** Make sure that you have completed all the steps in the [instructions](../README.md#virtual-machine-setup) for installing the VM & other course tools first.

Follow the [Project Submission Instructions](../README.md#submitting-projects) to upload your report from the course VM to [Bonnie](https://bonnie.udacity.com) using the `omscs` CLI.

  - Submit your report with the CLI on the VM from within the project folder:

```
$ omscs submit report
```

**Note:** The total size of your project (report + resources) must be less than 8MB for this project. If your submission is too large, you can reduce the scale of your images or report.


## Evaluation Criteria

Your submission will be graded based on:

  - Submission of an appropriate photograph
  - Thoughtful and appropriate answers to all questions in the template (Use complete sentences!)
  - Correctly extracting and reporting EXIF data from the photograph
  - Insights based on Intro lecture on Photography and Computational Photography as applied to this photographic artifact.
