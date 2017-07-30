# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import face_recognition
from datetime import datetime
from moviepy.editor import VideoFileClip

file_name  = 'data/video_11.mp4'

clip  = VideoFileClip(file_name) # can be gif or movie
# Main

# Frames in Video
for frame in clip.iter_frames(fps=30):
    # shape of image
  gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Converto to Color Image
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Find all the faces in the image
  face_locations = face_recognition.face_locations(frame)

  # shape of image
  image_h, image_w, _ = np.shape(frame)
  # Faces in image
  for face_location in face_locations:

    # Localize face in image
    top, right, bottom, left = face_location
    w, h = right - left, bottom - top

    xw1 = max(int(left    - 0.4 * w), 0)
    yw1 = max(int(top     - 0.4 * h), 0)
    xw2 = min(int(right   + 0.4 * w), image_w - 1)
    yw2 = min(int(bottom  + 0.4 * h), image_h - 1)
    
    # Face encoding 218d
    cropped_face  = frame[yw1:yw2 + 1, xw1:xw2 + 1]

    cv2.rectangle(frame, (xw1, yw1), (xw2, yw2), (255, 255, 255), 2, cv2.LINE_AA)
    
  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1) & 0xFF
 
  # if the `q` key was pressed, break from the loop
  if key == ord("q"):
    break

# do a bit of cleanup
cv2.destroyAllWindows()
