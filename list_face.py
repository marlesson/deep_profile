import numpy as np
from collections import Counter

class Face:
  def __init__(self, p1, p2, gender_value, gender, age):
    self.p1 = p1
    self.p2 = p2
    self.gender = gender
    self.gender_value = gender_value
    self.age    = age


  def is_intersection(self, face2):
    return (self.intersection_area(face2) > 0)
    # f1_left, f1_top     = self.p1
    # f1_right, f1_bottom = self.p2

    # f2_left, f2_top     = face2.p1
    # f2_right, f2_bottom = face2.p2

    # left    = max(f1_left, f2_left);
    # top     = max(f1_top, f2_top);

    # right   = min(f1_right, f2_right);
    # bottom  = min(f1_bottom, f2_bottom);

    # return ((right - left) > 0 and (bottom - top) > 0)

  def intersection_area(self, face2):
    f1_left, f1_top     = self.p1
    f1_right, f1_bottom = self.p2

    f2_left, f2_top     = face2.p1
    f2_right, f2_bottom = face2.p2

    left    = max(f1_left, f2_left);
    top     = max(f1_top, f2_top);

    right   = min(f1_right, f2_right);
    bottom  = min(f1_bottom, f2_bottom);

    if ((right - left) > 0 and (bottom - top) > 0):
      return (right - left) * (bottom - top)
    else:
      return 0

class ListFace:
  def __init__(self):
    self.faces  = []
    self.frames = 0

  def add_face(self, face):
    self.frames += 1
    if len(self.faces) > 500:
      self.faces.pop(0)
      self.faces.append(face)
    else:
      self.faces.append(face)

  # Test if face intersection with last face
  def is_intersection(self, face):
    if len(self.faces) > 0:
      return self.faces[-1].is_intersection(face)
    else:
      return False

  # Returns the areaf of rectangle that forms in the intersection
  def intersection_area(self, face):
    if len(self.faces) > 0:
      return self.faces[-1].intersection_area(face)
    else:
      return 0

  def avg_age(self):
    return int(np.mean([f.age for f in self.faces]))

  def avg_gender(self):
    b = Counter([f.gender for f in self.faces])
    return b.most_common(1)[0][0]

  def avg_gender_value(self):
    return np.mean([f.gender_value for f in self.faces])

  def avg_gender_name(self):
    return "Mulher" if self.avg_gender() == 'M' else "Homem"

  def gender_confidence(self):
    avg_value = self.avg_gender_value()

    if avg_value >= 0.5:
      suspected = (1 - avg_value)
    else:
      suspected = avg_value

    return 1 - suspected

  def inspection(self):
    return "{} ({}%), {} anos".format(self.avg_gender_name(), int(self.gender_confidence()*100), self.avg_age())
