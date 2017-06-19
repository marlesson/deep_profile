import os
import cv2
import dlib
import numpy as np
import time
import imutils
from imutils.video import VideoStream
from imutils import face_utils
from deep_profile import DeepProfile

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

def main():
    dp  = DeepProfile()
    vs  = VideoStream(resolution=(FRAME_WIDTH, FRAME_HEIGHT)).start()  #
    time.sleep(2.0)

    while True:
        # get video frame
        frame = vs.read()
        frame = imutils.resize(frame, width=FRAME_WIDTH)

        dp.profiles_image(frame)

        cv2.imshow("result", frame)
        
        key = cv2.waitKey(30)

        if key == 27:
            break

if __name__ == '__main__':
    main()
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()  