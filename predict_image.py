import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib
import argparse


from deep_profile import DeepProfile

path  = 'data/images/'
files = [f for f in os.listdir(path) if os.path.isfile(path+f)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Just an example",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--sleep",type=float, help="Sleep Frame", default=2.0)
    parser.add_argument("--ignore-gender", action="store_true", help="skip files that exist")
    args = parser.parse_args()
    config = vars(args)
    print(config)

    dp    = DeepProfile(ignore_gender=config["ignore_gender"])

    for f in files:
      if ".jpg" in f:
        print(path+f)

        img = cv2.imread(path+f)
        
    #   dp.reset()
        dp.profiles_image(img)

        cv2.imwrite(path+'pred/'+f, img)
