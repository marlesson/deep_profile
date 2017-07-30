import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import cv2
import dlib
import numpy as np
from datetime import datetime

from list_face import ListFace 
from list_face import Face
#import face_recognition

from wide_resnet import WideResNet


class DeepProfile:
  def __init__(self):
    self.file_log = open("log/file_log", "a")

    self.img_size = 64
    self.model    = WideResNet(self.img_size, depth=16, k=8)()
    self.model.load_weights(os.path.join("pretrained_models", "weights.18-4.06.hdf5"))
    
    self.image     = None
    self.detector  = dlib.get_frontal_face_detector()
    self.list_face = []
    self.logs      = []

  def profiles_image(self, image):
    self.logs   = []
    self.image  = image

    # Detecte faces in image
    faces_detected = self.detecte_faces(self.image)

    # Matrix to faces in image
    faces = np.empty((len(faces_detected), self.img_size, self.img_size, 3))

    # Print retaatangle in face and get face in 64x64
    for i, d in enumerate(faces_detected):
        faces[i,:,:,:] = self.copy_face_in_image(self.image, d) 

    # Predict ages and genders of the detected faces
    result, genders, ages = self.predict(faces)
    if result:
      for i, d in enumerate(faces_detected):
          face_encode = self.face_encode(self.image, d)
          l_face      = self.add_list_face(self.image, d, genders[i], ages[i])
          
          self.draw_box(self.image, d, l_face.avg_gender(), l_face.avg_age())
          #self.draw_box(self.image, d, genders[i], ages[i])      

      self.logs.append("Face: {}".format(len(self.list_face)))
      
      for l_face in self.list_face:
        self.logs.append("Face -> : {}".format(len(l_face.faces)))

          

    self.display_log(self.image)
    return self.image

  def add_list_face(self, image, face, gender, ages):
    x1, y1, x2, y2, w, h = face.left(), face.top(), face.right() + 1, face.bottom() + 1, face.width(), face.height()
    

    # Face
    face = Face([x1, y1], [x2, y2], gender, int(ages))
    now_list_face = None

    # find list face of face
    for l_face in self.list_face:
      if l_face.is_intersection(face):
        now_list_face = l_face
        break

    # if not found
    if now_list_face == None:
      now_list_face = ListFace() #new 
      self.list_face.append(now_list_face)

    now_list_face.add_face(face)

    return now_list_face

  def predict(self, faces):
    results = self.model.predict(faces)
    if len(results) > 0:

      # Gender
      predicted_genders = [self.get_gender(g[0]) for g in results[0]]

      # Age
      ages      = np.arange(0, 101).reshape(101, 1)
      pred_ages = results[1].dot(ages).flatten()
      pred_ages = [int(age) for age in pred_ages]
      
      return True, predicted_genders, pred_ages
    else:
      return False, [], []

  def get_gender(self, value):
    return "M" if value >= 0.4 else "H"

  def detecte_faces(self, image):
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # detect faces using dlib detector
    detected = self.detector(input_image, 1)

    return detected

  def copy_face_in_image(self, image, face):
    image_h, image_w, _ = np.shape(image)

    x1, y1, x2, y2, w, h = face.left(), face.top(), face.right() + 1, face.bottom() + 1, face.width(), face.height()
    xw1 = max(int(x1 - 0.4 * w), 0)
    yw1 = max(int(y1 - 0.4 * h), 0)
    xw2 = min(int(x2 + 0.4 * w), image_w - 1)
    yw2 = min(int(y2 + 0.4 * h), image_h - 1)
    
    return cv2.resize(image[yw1:yw2 + 1, xw1:xw2 + 1, :], (self.img_size, self.img_size))

  def face_encode(self, image, face):
    image_h, image_w, _ = np.shape(image)

    x1, y1, x2, y2, w, h = face.left(), face.top(), face.right() + 1, face.bottom() + 1, face.width(), face.height()
    xw1 = max(int(x1 - 0.4 * w), 0)
    yw1 = max(int(y1 - 0.4 * h), 0)
    xw2 = min(int(x2 + 0.4 * w), image_w - 1)
    yw2 = min(int(y2 + 0.4 * h), image_h - 1)

    cropped_face  = image[yw1:yw2 + 1, xw1:xw2 + 1]
    face_encoding = np.ones(5)#face_recognition.face_encodings(cropped_face)[0]

    return face_encoding

  def draw_rectangle(self, image, point1, point2, color):
    cv2.rectangle(image, point1, point2, color, 2, cv2.LINE_AA)

  def draw_box(self, image, face, gender, ages):
    x1, y1, x2, y2, w, h = face.left(), face.top(), face.right() + 1, face.bottom() + 1, face.width(), face.height()
    
    if gender == 'M':
      color = (0, 0, 255)
    else:
      color = (255, 0, 0)  
    
    self.draw_label(image, face, gender, ages, color)
    self.draw_rectangle(image, (x1, y1), (x2, y2), color)

  def draw_label(self, image, face, gender, ages, color,
                 font=cv2.FONT_HERSHEY_DUPLEX,
                 font_scale=0.5, thickness=1):
    
    point = (face.left(), face.top())
    image_h, image_w, _ = np.shape(image)

    label2 = "{}".format("Mulher" if gender == 'M' else "Homem")
    label1 = "[{}]".format(ages)
    
    #size  = max(cv2.getTextSize(label1, font, font_scale, thickness)[0], cv2.getTextSize(label2, font, font_scale, thickness)[0])
    size  = cv2.getTextSize(label1, font, font_scale, thickness)[0]
    x, y  = point
    y     -= 7
    x     -= 1
    
    cv2.rectangle(image, (x, y - size[1]-5), (x + size[0]+2, y+5), color, cv2.FILLED)
    cv2.putText(image, label1, (x, y), font, font_scale, (255, 255, 255), thickness)

  def display_log(self, image, font=cv2.FONT_HERSHEY_DUPLEX,
                 font_scale=0.5, thickness=1):
    x = 0
    y = 10

    for log in self.logs:
      cv2.putText(image, log, (x, y), font, font_scale, (255, 255, 255), thickness)
      y = y + cv2.getTextSize(log, font, font_scale, thickness)[0][1]

  def print_log(self, gender, ages, face_encode):
    now    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    face   = ",".join([str(i) for i in face_encode])
    output = "{}, {}, {}, {}\n".format(now, gender, ages, face)
    
    self.file_log.write(output)
    
