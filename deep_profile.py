import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


import cv2
import dlib
import numpy as np
from wide_resnet import WideResNet

class DeepProfile:
  def __init__(self):
    self.img_size = 64
    self.model = WideResNet(self.img_size, depth=16, k=8)()
    self.model.load_weights(os.path.join("pretrained_models", "weights.18-4.06.hdf5"))
    self.image = None
    self.detector = dlib.get_frontal_face_detector()


  def profiles_image(self, image):
    self.image  = image

    # Detecte faces in image
    faces_detected = self.detecte_faces(self.image)

    # Matrix to faces in image
    faces = np.empty((len(faces_detected), self.img_size, self.img_size, 3))

    # Print retaatangle in face and get face in 64x64
    for i, d in enumerate(faces_detected):
        faces[i,:,:,:] = self.copy_face_in_image(self.image, d) 

    #predict ages and genders of the detected faces
    results = self.predict(faces)
    if len(results) > 0:
      # Gender
      predicted_genders = results[0]

      # Age
      ages = np.arange(0, 101).reshape(101, 1)
      predicted_ages = results[1].dot(ages).flatten()

      # draw results
      for i, d in enumerate(faces_detected):
          gender = "M" if predicted_genders[i][0] >= 0.5 else "H"
          ages   = int(predicted_ages[i])
          self.draw_label(self.image, (d.left(), d.top()), gender, ages)

    return self.image

  def detecte_faces(self, image):
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect faces using dlib detector
    detected = self.detector(input_image, 0)

    return detected

  def predict(self, faces):
    results = self.model.predict(faces)
    return results

  def copy_face_in_image(self, image, face):
    image_h, image_w, _ = np.shape(image)

    x1, y1, x2, y2, w, h = face.left(), face.top(), face.right() + 1, face.bottom() + 1, face.width(), face.height()
    xw1 = max(int(x1 - 0.4 * w), 0)
    yw1 = max(int(y1 - 0.4 * h), 0)
    xw2 = min(int(x2 + 0.4 * w), image_w - 1)
    yw2 = min(int(y2 + 0.4 * h), image_h - 1)
    
    self.draw_rectangle(image, (x1, y1), (x2, y2))
    #self.draw_rectangle(image, (xw1, yw1), (xw2, yw2))
    
    return cv2.resize(image[yw1:yw2 + 1, xw1:xw2 + 1, :], (self.img_size, self.img_size))

  def draw_rectangle(self, image, point1, point2):
    cv2.rectangle(image, point1, point2, (255, 255, 255), 2, cv2.LINE_AA)

  def draw_label(self, image, point, gender, ages, 
                 font=cv2.FONT_HERSHEY_DUPLEX,
                 font_scale=0.5, thickness=1):
      
    image_h, image_w, _ = np.shape(image)

    label2 = "{}".format("Mulher" if gender == 'M' else "Homem")
    label1 = "Idade: {}".format(ages)
    
    size  = max(cv2.getTextSize(label1, font, font_scale, thickness)[0], cv2.getTextSize(label2, font, font_scale, thickness)[0])
    x, y  = point
    y     -= 9
    
    cv2.rectangle(image, (x, y - size[1]*2-5), (x + size[0]+2, y+5), (0,0,0), cv2.FILLED)
    
    cv2.putText(image, label1, (x, y), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(image, label2, (x, y-size[1]-5), font, font_scale, (255, 255, 255), thickness)
