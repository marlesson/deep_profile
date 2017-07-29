# Etapa 2.
# Complemento da primeira etapa
# Usa de DeepLearning para identificar sexo e idade
#
# data/video/face - imagens

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib
import json
import os.path

from deep_profile import DeepProfile
from moviepy.editor import VideoFileClip

path_data  = 'data/video_2'

def read_img(file):
  path = path_data+"/face/"+file
  return cv2.imread(path_data+"/face/"+file)

def save_json(file, genders, ages):
  with open(path_data+"/data/"+file, 'r+') as f:
    json_data = json.load(f)
    json_data['gender'] = genders[0]
    json_data['age']    = ages[0]    
    f.seek(0)
    f.write(json.dumps(json_data))
    f.truncate()

# Main
dp  = DeepProfile()

for file in os.listdir(path_data+"/data"):
  if ".json" in file:
    print(file)

    face  = read_img(file.replace("json", "jpg"))
    face  = cv2.resize(face, (64, 64))

    # Predict ages and genders of the detected faces
    result, genders, ages = dp.predict(np.expand_dims(face, axis=0))
    if result:
      save_json(file, genders, ages)