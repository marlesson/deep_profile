import os
import cv2
import dlib
import numpy as np
import time
import imutils
import matplotlib.pyplot as plt

from deep_profile import DeepProfile
from moviepy.editor import *

# Deep Profile
dp  = DeepProfile()


path  = 'data/'
files = [f for f in os.listdir(path) if os.path.isfile(path+f)]

dp    = DeepProfile()

for f in files:
  print(f)

  file_name = path+f
  if ".mp4" in f:
    dp.reset()
    
    # Play
    clip = VideoFileClip(file_name)    

    # Process video with 1 fps
    for frame in clip.iter_frames(fps=1):
        # Process frame
        dp.profiles_image(frame)

    print(file_name)
    # Print Result
    for i, face in enumerate(dp.list_face):
        print(i, " ", face.inspection())

    print()

    dp.reset()
    # Build a processed video
    new_clip = clip.fl_image(dp.profiles_image)
    new_clip.write_videofile("data/proc/"+f) 