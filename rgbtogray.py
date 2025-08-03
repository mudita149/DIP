import cv2

image = cv2.imread(r"C:\Users\Mudita Shukla\Desktop\coding\pythonarduino\earth.jpg")

if image is None:
    print(" Failed to load image.")
else:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray_image)
    cv2.imwrite(r'C:\Users\Mudita Shukla\Desktop\coding\pythonarduino\output_gray.jpg', gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()