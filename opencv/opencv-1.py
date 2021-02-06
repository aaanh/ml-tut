import cv2

# Read image
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
# Show image
cv2.imshow("pic", image)
# Pause screen
cv2.waitKey()
# Close all windows
cv2.destroyAllWindows()
