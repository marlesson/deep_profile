import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib


from deep_profile import DeepProfile

path  = 'data/'
files = [f for f in os.listdir(path) if os.path.isfile(path+f)]

dp    = DeepProfile()

for f in files:
    img = cv2.imread(path+f)
    print(path+f)
    dp.profiles_image(img)

    #cv2.imshow('image',img)
    cv2.imwrite(path+'pred/'+f, img)
