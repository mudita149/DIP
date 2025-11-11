# ==============================
# EDGE DETECTION USING CONVOLUTION (from scratch)
# Kernel Used: Laplacian Operator
# ==============================

import numpy as np
from PIL import Image

# Step 1️⃣: Load the image and convert it to grayscale
# ------------------------------------------
# "L" mode converts the image to 8-bit grayscale (values 0–255)
img = Image.open("sample.jpg").convert("L")

# Convert PIL image to a NumPy array for pixel-wise processing
a = np.array(img)

# Display the shape (height, width) of the grayscale image
print("Image shape:", a.shape)

# Step 2️⃣: Define the convolution kernel (Laplacian)
# ------------------------------------------
# This kernel highlights areas of rapid intensity change (edges)
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Extract kernel dimensions
kh, kw = kernel.shape

# Calculate padding size (for a 3x3 kernel, pad = 1)
pad = kh // 2

# Step 3️⃣: Pad the image to handle border pixels
# ------------------------------------------
# Padding adds an extra border of zeros so the kernel can slide over edges properly
p = np.pad(a, ((pad, pad), (pad, pad)), mode='constant')

# Step 4️⃣: Prepare an empty output array
# ------------------------------------------
# Same size as the original image, using float32 for accuracy during computation
out = np.zeros_like(a, dtype=np.float32)

# Step 5️⃣: Perform convolution manually using nested loops
# ------------------------------------------
# For each pixel, multiply the kernel with its corresponding neighborhood region
# and take the sum of the products to get the edge strength
h, w = a.shape
for y in range(h):
    for x in range(w):
        # Extract the region of the image covered by the kernel
        region = p[y:y+kh, x:x+kw]
        # Element-wise multiplication and summation
        out[y, x] = np.sum(region * kernel)

# Step 6️⃣: Normalize and clip pixel values
# ------------------------------------------
# Values may exceed [0,255] or go negative after convolution, so clip them
out = np.clip(out, 0, 255).astype(np.uint8)

# Step 7️⃣: Save and display the result
# ------------------------------------------
# Convert NumPy array back to an image format and save it
res = Image.fromarray(out)
res.save("output.jpg")

# Display the resulting edge-detected image
res.show()

print("Edge detection complete. Output saved as 'output_convolved.jpg'")
