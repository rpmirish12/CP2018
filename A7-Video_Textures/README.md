# Video Textures

## Synopsis

Purpose of creating [video textures](http://www.cc.gatech.edu/cpl/projects/videotexture/) (infinitely looping pieces of video). These are basically gifs with very smooth transtitions. The process is also described in [Video Textures](http://cs.colby.edu/courses/F07/cs397/papers/schodl-videoTextures-sig00.pdf) (Scholdl, et al; SIGGRAPH 2000).


## Instructions

### 1. Implement the functions in the `textures.py` file.

- `videoVolume`: Take a list containing image numpy arrays, and turn them into a single array which contains the entire video volume.
- `computeSimilarityMetric`: Find the "distance" between every pair of frames in the video.
- `transitionDifference`: Incorporate frame transition dynamics into a difference matrix created by computeSimilarityMetric.
- `findBiggestLoop`: Find an optimal loop for the video texture. (NOTE: part of your task is to determine the best value for the alpha parameter.)
- `synthesizeLoop`: Take our video volume and turn it back into a series of images, keeping only the frames in the loop you found. 

The docstrings of each function contain detailed instructions.

**Notes:**

- The `main.py` script reads files in the sorted order of their file name according to the conventions of python string sorting; it is essential that file names are chosen so that they are in sequential order. 

#### Finding a good alpha
The last bit of computation is the alpha parameter, which is a scaling factor. The size of the loop and the transition cost are likely to be in very different units, so we introduce a new parameter to make them comparable. We can manipulate alpha to control the tradeoff between loop size and smoothness. Large alphas prefer large loop sizes, and small alphas bias towards short loop sizes. You are looking for an alpha between these extremes (the goldilocks alpha). Your findBiggestLoop function has to compute this score for every choice of start and end, and return the start and end frame numbers that corresponds to the largest score. 

More than one frame, and less than all of the frames. Alpha may vary significantly (by orders of magnitude) for different input videos.  When your coding is complete the main.py program will generate visualization images of the three difference matrices: similarity, transition, and scoring.  These matrices may help in identifying good alpha values.

## Appendix - Working with Video

Working with video is not always user friendly. It is difficult to guarantee that a particular video codec will work across all systems. In order to avoid such issues, the inputs for this assignment are given as a sequence of numbered images.

**ffmpeg (avconv)**
These are free and very widely used software for dealing with video and audio.

- ffmpeg is available [here](http://www.ffmpeg.org/)
- avconv is available [here](https://libav.org/avconv.html)

Example ffmpeg Usage:

You can use this command to split your video into frames:
```ffmpeg -i video.ext -r 1 -f image2 image_directory/%04d.png```

And this command to put them back together:
```ffmpeg -i image_directory/%04d.png out_video.gif```
