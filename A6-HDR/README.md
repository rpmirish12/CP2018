# High Dynamic Range Imaging

## Synopsis

Focus on the core algorithms behind computing HDR images based on the paper “Recovering High Dynamic Range Radiance Maps from Photographs” by Debevec & Malik (available in the course resources on T-Square). The notational conventions & overall structure is explained in the paper.


## Instructions

### 1. Implement the functions in the `hdr.py` file.

- `linearWeight`: Determine the weight of a pixel based on its intensity.
- `sampleIntensities`: Randomly sample pixel intensity exposure slices for each possible pixel intensity value from the exposure stack.
- `computeResponseCurve`: Find the camera response curve for a single color channel by finding the least-squares solution to an overdetermined system of equations.
- `computeRadianceMap`: Use the response curve to calculate the radiance map for each pixel in the current color layer.

The docstrings of each function contains detailed instructions.

*Notes*:
- Images in the `images/source/sample` directory are provided for testing

- It is essential to put  images in exposure order and name them in this order, similar to the input/sample images. For the given sample images of the home, the exposure info is given in main.py and repeated here (darkest to lightest):
`EXPOSURE TIMES = np.float64([1/160.0, 1/125.0, 1/80.0, 1/60.0, 1/40.0, 1/15.0])`

- Image alignment is critical for HDR. Ensure that your images are aligned and cropped to the same dimensions
