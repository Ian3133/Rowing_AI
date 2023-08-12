import cv2
import mediapipe as mp
import numpy as np

drawing = mp.solutions.drawing_utils
body = mp.solutions.pose      

cam = cv2.VideoCapture(0) 
# should play around with what percent both at 50%
with body.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cam.isOpened():
        ret, img = cam.read()
        
        img.flags.writeable = False
        results = pose.process(img)
        img.flags.writeable = True  # Set back to True to draw on the image
        
        if results.pose_landmarks:
            drawing.draw_landmarks(img, results.pose_landmarks, body.POSE_CONNECTIONS,
                                   drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
                                   drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                  )
        
        
        cv2.imshow("Footie", img)
        
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
cam.release()
cv2.destroyAllWindows()
