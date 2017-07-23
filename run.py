#from deep_profile import DeepProfile
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib
import face_recognition
import json

from moviepy.editor import VideoFileClip

file_name  = 'video_1.mp4'
path_data  = 'data/video_1'

clip  = VideoFileClip(file_name) # can be gif or movie
count_frame = 1
count_face  = 1

def save_face(file, face_image):
  cv2.imwrite('{}/face/{}.jpg'.format(path_data, file), face_image)

def save_json(file, frame, face_128_dim):
  face = ",".join([str(i) for i in face_128_dim])
  data = {'frame': frame, 'filename': file, 'face_encode': face_128_dim.tolist()}
  
  with open('{}/data/{}.json'.format(path_data, file), 'w') as outfile:
    json.dump(data, outfile)

# frames = int(clip.fps * clip.duration)
# print(clip.duration)
# print(clip.fps)

# Frames in Video
for frame in clip.iter_frames(fps=1):
    # shape of image
    image_h, image_w, _ = np.shape(frame)

    # Converto to Color Image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces in the image
    face_locations = face_recognition.face_locations(frame)

    print("{}: I found {} face(s) in this photograph.".format(count_frame, len(face_locations)))

    # Faces in image
    for face_location in face_locations:

        # Localize face in image
        top, right, bottom, left = face_location
        w, h = right - left, bottom - top

        xw1 = max(int(left    - 0.4 * w), 0)
        yw1 = max(int(top     - 0.4 * h), 0)
        xw2 = min(int(right   + 0.4 * w), image_w - 1)
        yw2 = min(int(bottom  + 0.4 * h), image_h - 1)

        # You can access the actual face itself like this:
        face_image = frame[yw1:yw2 + 1, xw1:xw2 + 1]

        # File name with frame and face
        file = "Frame_{}_Face_{}".format(count_frame, count_face)

        # FaceToVec
        face_128_dim = face_recognition.face_encodings(face_image)
    
        if len(face_128_dim) > 0:
            # Save /face
            save_face(file, face_image)
            # Save /data
            save_json(file, count_frame, face_128_dim[0])

            count_face = count_face + 1
    count_frame = count_frame + 1

    # if count_frame > 5:
    #   break 