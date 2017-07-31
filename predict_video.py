import os
import cv2
import dlib
import numpy as np
import time
import imutils

from deep_profile import DeepProfile
from moviepy.editor import VideoFileClip

file_name  = 'data/video_16.mp4'


def main():
    dp  = DeepProfile()

    clip  = VideoFileClip(file_name) # can be gif or movie
    # Main

    # Frames in Video
    for frame in clip.iter_frames(fps=60):
        # shape of image
        image_h, image_w, _ = np.shape(frame)

        # Converto to Color Image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        dp.profiles_image(frame)

        cv2.imshow("result", frame)
        
        key = cv2.waitKey(30)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

if __name__ == '__main__':
    main()
    # do a bit of cleanup
    cv2.destroyAllWindows()
