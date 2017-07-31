import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib


from deep_profile import DeepProfile

path  = 'data/images/'
files = [f for f in os.listdir(path) if os.path.isfile(path+f)]

dp    = DeepProfile()

#for f in files:
for i in range(0, 15):
  f = "familia-feliz-face.jpg"
  if ".jpg" in f:
    print(path+f)

    img = cv2.imread(path+f)
    
 #   dp.reset()
    dp.profiles_image(img)

    cv2.imwrite(path+'pred/'+f, img)
