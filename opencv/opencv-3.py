import cv2

image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("image_gray.jpg", image)
