import os
import cv2
import dlib
import numpy as np
import time
import imutils
import argparse

from imutils.video import VideoStream
from imutils import face_utils
from deep_profile import DeepProfile

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

def main(params):
    dp  = DeepProfile(
        ignore_gender=params["ignore_gender"]
    )
    vs  = VideoStream(resolution=(FRAME_WIDTH, FRAME_HEIGHT)).start()  #
    time.sleep(float(params["sleep"]))

    while True:
        # get video frame
        frame = vs.read()
        frame = imutils.resize(frame, width=FRAME_WIDTH)

        dp.profiles_image(frame)

        cv2.imshow("result", frame)
        
        key = cv2.waitKey(30)

        if key == 1048690:
            dp.reset()

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Just an example",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--sleep",type=float, help="Sleep Frame", default=2.0)
    parser.add_argument("--ignore-gender", action="store_true", help="skip files that exist")
    args = parser.parse_args()
    config = vars(args)
    print(config)

    main(config)
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()  