import cv2

image = cv2.imread(r"C:\Users\Mudita Shukla\Desktop\coding\pythonarduino\earth.jpg")

if image is None:
    print(" Failed to load image.")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Black and White", bw_image)
    cv2.imwrite(r"C:\Users\Mudita Shukla\Desktop\coding\pythonarduino\output_bw.jpg", bw_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
