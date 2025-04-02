
import cv2
import numpy as np
from cv2 import aruco

# get dicionary and get parameters
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

cap = cv2.VideoCapture(2)
while(cap.isOpened):
    ret, frame = cap.read()
    if True == ret:
        corners, ids, rejectedCandidates = aruco.detectMarkers(frame, dictionary, parameters=parameters)
        print(ids)
        #print(corners)
        aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow("webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()