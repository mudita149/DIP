import cv2

img = cv2.imread(r"C:\Users\Mudita Shukla\Desktop\coding\pythonarduino\earth.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError("Image not found.")

equalized_img = cv2.equalizeHist(img)

cv2.imshow("Original Image", img)
cv2.imshow("Histogram Equalized Image", equalized_img)
cv2.imwrite(r"C:\Users\Mudita Shukla\Desktop\coding\pythonarduino\histogram_output.jpg", equalized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

