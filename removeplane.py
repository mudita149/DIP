import cv2
import numpy as np

image = cv2.imread(r"C:\Users\Mudita Shukla\Desktop\coding\pythonarduino\earth.jpg")

if image is None:
    print(" Failed to load image.")
else:
    blue_only = image.copy()
    blue_only[:, :, 1] = 0
    blue_only[:, :, 2] = 0
    cv2.imshow("Only Blue Plane", blue_only)
    cv2.imwrite(r"C:\Users\Mudita Shukla\Desktop\coding\pythonarduino\earth_output_blue_only.jpg", blue_only)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
