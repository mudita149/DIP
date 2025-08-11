Experiment 2A – Histogram Equalization
📌 Description
Histogram equalization is a core image processing technique for enhancing contrast. It redistributes the most frequent pixel intensity values, effectively expanding the image’s dynamic range.

Imagine a dark photograph where most pixel values are clustered in the shadows—histogram equalization spreads these values evenly across the full intensity scale (0–255), revealing hidden details. The goal is to produce a more uniform (or “flat”) histogram.

🎯 Objectives
Understand the concept of an image histogram and its effect on contrast.

Implement histogram equalization on a grayscale image using Python’s OpenCV.

Compare the original and equalized images (and their histograms) to visualize the improvement.

🖼️ Sample Output
The equalized image reveals more detail in both dark and bright regions.

The histogram spreads across the full range instead of being tightly clustered.

Experiment 2B – Bit-Plane Slicing in Digital Image Processing
📌 Description
This experiment demonstrates bit-plane slicing on a grayscale image using OpenCV and Matplotlib. Bit-plane slicing isolates individual bits in each pixel’s intensity value, which is useful for compression, enhancement, and feature extraction.

A grayscale pixel (0–255) is represented by 8 bits. Each bit contributes differently to the image’s appearance—the higher bits hold more significant details.

🎯 Objectives
Understand and implement bit-plane slicing in Python.

Extract all 8 bit-planes using vectorized operations.

Visualize the contribution of each bit to image quality.

⚙️ Implementation Logic
Read the grayscale image with cv2.imread().

Create an empty list to store the 8 bit-plane images.

For each bit i from 0 to 7:

Create a mask: mask = 2**i

Isolate the bit using a bitwise AND: bit_plane = img & mask

Normalize: set non-zero values to 255 (white)

Append to the bit-plane list

Use Matplotlib to display the original image and all 8 bit-planes in a subplot grid.

🖼️ Sample Output
Bit-plane 7 (MSB): Shows the most significant details, resembling the original image.

Bit-plane 0 (LSB): Shows the least significant details, often appearing as random noise.
