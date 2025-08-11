import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\Mudita Shukla\Desktop\coding\pythonarduino\earth.jpg", cv2.IMREAD_GRAYSCALE)


if img is None:
    raise FileNotFoundError("Image not found. Make sure 'image.jpg' is in the working directory.")

bit_planes = []
for i in range(8):
    bit_plane = cv2.bitwise_and(img, (1 << i))
    bit_plane = np.where(bit_plane > 0, 255, 0).astype(np.uint8)
    bit_planes.append(bit_plane)

plt.figure(figsize=(12, 6))
plt.subplot(3, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

for i in range(8):
    plt.subplot(3, 3, i+2)
    plt.imshow(bit_planes[i], cmap='gray')
    plt.title(f'Bit Plane {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()
