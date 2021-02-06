# Read from camera device

import cv2

camera_id = 0

# Open camera
cap = cv2.VideoCapture(camera_id)

# Read img from camera
while(True):
    ret, frame = cap.read()
    cv2.imshow("cam_0", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
cap.release()
cv2.destroyAllWindows()
