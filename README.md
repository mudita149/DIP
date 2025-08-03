# DIP
DIP lab codes 
Practical 1(a) --> To convert the given image from rgb to gray
Purpose: Converts a color image to grayscale.

How it works:
Uses cv2.cvtColor() with the COLOR_BGR2GRAY flag to compute intensity from RGB.

Output: Displays the grayscale image and saves it.

1(b) --> To work on different planes and extract the blue plane from the colored image
Purpose: Extracts the blue color channel from a color image.

How it works:
Sets the green and red channels to zero, keeping only the blue.

Output: Displays and saves a new image showing only blue components.

1(c) --> To convert rgb to black and white
Purpose: Converts an image to black and white (binary) using thresholding.

How it works:
Converts the image to grayscale, then applies a fixed threshold (127) to produce a binary image (0 or 255 pixel values).

Output: Displays and saves a high-contrast black and white version.
