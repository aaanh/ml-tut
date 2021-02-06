import cv2

# Read greyscale and convert

image = cv2.imread("image.jpg")
cv2.imshow("Color", image)
cv2.waitKey()

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("BW", image_gray)
cv2.waitKey()

cv2.destroyAllWindows()
