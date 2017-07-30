import numpy as np
from collections import Counter

class Face:
  def __init__(self, p1, p2, gender, age):
    self.p1 = p1
    self.p2 = p2
    self.gender = gender
    self.age    = age

  def is_intersection(self, face2):
    f1_left, f1_top     = self.p1
    f1_right, f1_bottom = self.p2

    f2_left, f2_top     = face2.p1
    f2_right, f2_bottom = face2.p2

    left    = max(f1_left, f2_left);
    top     = max(f1_top, f2_top);

    right   = min(f1_right, f2_right);
    bottom  = min(f1_bottom, f2_bottom);

    return ((right - left) > 0 and (bottom - top) > 0)

class ListFace:
  def __init__(self):
    self.faces = []

  def add_face(self, face):
    if len(self.faces) > 30:
      self.faces.pop(0)
      self.faces.append(face)
    else:
      self.faces.append(face)

  def is_intersection(self, face):
    if len(self.faces) > 0:
      return self.faces[-1].is_intersection(face)
    else:
      return False

  def avg_age(self):
    return int(np.mean([f.age for f in self.faces]))

  def avg_gender(self):
    b = Counter([f.gender for f in self.faces])
    return b.most_common(1)[0][0]

