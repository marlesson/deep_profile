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

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "pretrained_models/shape_predictor_68_face_landmarks.dat"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True,
 # help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
  help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_model)#args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

file_log = open("log/file_log_face", "a")

def print_log(file_log, face_encode):
  now    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  face   = ",".join([str(i) for i in face_encode])
  output = "{}, {}\n".format(now, face)
  
  file_log.write(output)

# loop over the frames from the video stream
while True:
  # grab the frame from the threaded video stream, resize it to
  # have a maximum width of 400 pixels, and convert it to
  # grayscale
  frame = vs.read()
  frame = imutils.resize(frame, width=400)
  gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
  # detect faces in the grayscale frame
  rects = detector(gray, 0)

  for i, d in enumerate(rects):
    image_h, image_w, _ = np.shape(frame)

    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
    xw1 = max(int(x1 - 0.4 * w), 0)
    yw1 = max(int(y1 - 0.4 * h), 0)
    xw2 = min(int(x2 + 0.4 * w), image_w - 1)
    yw2 = min(int(y2 + 0.4 * h), image_h - 1)
    
    # Face encoding 218d
    cropped_face  = frame[yw1:yw2 + 1, xw1:xw2 + 1]
    #face_encoding = face_recognition.face_encodings(cropped_face)[0]
    #print_log(file_log, face_encoding)

    # Print retangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
    
  # loop over the face detections
  for rect in rects:
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
 
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
      cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    
  # show the frame
  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1) & 0xFF
 
  # if the `q` key was pressed, break from the loop
  if key == ord("q"):
    break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()  