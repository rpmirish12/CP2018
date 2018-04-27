# Epsilon Photography

**Important Note:** This assignment is subject to the "Above & Beyond" rule. In summary: meeting all stated requirements will earn 90%; the last 10% is reserved for individual effort to research, implement, and report some additional high-quality work on this topic beyond the minimum requirements. Your A&B work must be accompanied by discussion of the computational photographic concepts involved, and will be graded based on the level of effort, and quality of your results and documentation in the report. (Please review the full explanation of this rule in the syllabus or on Piazza.)


## Synopsis

Epsilon photography is a form of computational photography wherein multiple images are captured by varying a camera parameter such as aperture, exposure, focus, film speed, viewpoint, etc., by a small amount ε (hence the name) for the purpose of enhanced post-capture flexibility. (Wikipedia definition) The most critical aspect of epsilon photography is that *only one parameter changes* throughout the image sequence. 

For example, you may capture multiple images of a *stationary scene*:

  - At a specified interval between frames (e.g., a time-lapse)
  - Under different lighting conditions (e.g., flash/no-flash)
  - From different viewpoints
  - At different exposures
  - With different focal planes
  - At different apertures
  - With a *single* moving subject (e.g., stop motion with a single subject)
  - Many more! Be creative, but please remember the goals and restrictions of the assignment.

**Note:** You may NOT do HDR photography or a Panorama for this assignment -- we will work with both of these topics in their own assignments later.

Some examples of Epsilon Photography from the past that went "above and beyond" 

  - Pictures of shaving one’s head, merged the aligned head pictures with different hairs to showcase an amazing final picture;
  - Pictures that were used to generate light art (remember: small changes, and only change *one* thing)
  - Time lapse of traffic, a flower opening, a snail crawling (ok, still haven’t seen a snail)
  - Times of the day, with an artifact that artfully blended the changing lighting across the artifact
  - A great final product, where the student wrote code to put all the images together

We look forward to adding more here too!


What NOT to do? Some examples from former students!

  - Do NOT submit pictures of your dog (pet) wearing different clothes and jumping around (maybe an example of epsilon fashionography, but not what we want!).
  - Do NOT submit one picture from each of your last 4 vacations. Vacation Photography, sure, but not Epsilon Photography.
  - Do NOT submit blurry stills extracted from a video taken with your smartphone.
  - Do NOT submit separate pictures of each member of your family in the same pose.
  - Do NOT submit pictures with light art where there are multiple changes (scene, light color, etc.)
  - Do NOT submit a series of pictures where each one has a _different_ small change (1st is exposure, 2nd is ISO, .... )

We are looking for evidence of intentional planning in your images. We want to see you plan what the images will be and how you can generate a new or novel view or image. Think about what your camera can do currently and how you can use computation to merge the resulting images into a novel result.


## Instructions


### 1. Create a sequence of 4-8 images using epsilon photography (we will refer to these as your "N pictures").

Your N pictures should have almost everything in common, and only **ONE thing varying** in very small (epsilon) amounts. (This may be challenging.) You should vary **ONLY one parameter** across all the images, not all parameters. And again, that parameter should vary in the smallest amount from one picture to another.

Why do we repeat that it should be **only ONE parameter changing** so many times? Because misunderstanding this basic goal is the most common deduction on this assignment. It is critical that only one element changes. Any simultaneous changes in the scene, camera parameters, or camera orientation is not epsilon photography.

Most people will find they need a way to hold their camera still for this project, because holding the camera by hand rarely works well for keeping a static scene; using a support (improvised or tripod) generally produces better results. Small and/or cheap tripods exist, but are not required. (Although you may find them useful on later assignments, too.)

You may use any source code, open source tool, or commercial software (including Gimp, Photoshop, or other package software) to help remove minor camera shake by stacking and cropping, but you must note that you did this in your report and mention which software you used.


### 2. Create a final artifact

Combine your image sequence to produce a novel photographic artifact. What can you do with these N images to generate a new image that shows a novel view or representation?  Be creative. Use whatever code or tool (commercial or open source package) that you find necessary to create your final artifact. (GIFs are fine if they demonstrate your epsilon change.)

Add your N images and final artifact to a file named `resources.zip` in the project folder.


### 3. Complete the Report

Make a copy of the [report template](https://drive.google.com/open?id=1HwJJc7Jn2FMlJIT1Oq_OUGb2is0eoIeiM1h6IQlFhxA) and answer all of the questions. Save your report as `report.pdf` in the project directory. 


### 4. Submit the Project

Make sure that files named `report.pdf` and `resources.zip` containing your N images and final artifact are in the assignment folder -- both files will be uploaded. (These names are important because the submission script searches the current working directory for exact matches to the filenames). Your images must be one of the following types: jpg, jpeg, bmp, png, tif, or tiff.

**Note:** Make sure that you have completed all the steps in the [instructions](../README.md#virtual-machine-setup) for installing the VM & other course tools first.

Follow the [Project Submission Instructions](../README.md#submitting-projects) to upload your report from the course VM to [Bonnie](https://bonnie.udacity.com) using the `omscs` CLI.

  - Submit your report with the CLI on the VM from within the project folder:

```
$ omscs submit report
```

**Note:** The total size of your project (report + resources) must be less than 30MB for this project. If your submission is too large, you can reduce the scale of your images or report.


## Criteria for Evaluation

Your submission will be graded based on:

  - Creativity, choice of domain, result quality, and a workflow that demonstrates Computational Photography. 
  - Explaining your thought process as it relates to controlling your epsilon parameter change

If you turn in something with significant variations in each image or more than one change between frames, then you have NOT followed the instructions and the goal of this assignment and points will be deducted. 
