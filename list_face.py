import numpy as np
from collections import Counter

class Face:
  def __init__(self, p1, p2, gender, age):
    self.p1 = p1
    self.p2 = p2
    self.gender = gender
    self.age    = age

  def is_intersection(self, face2):
    return True

class ListFace:
  def __init__(self):
    self.faces = []

  def add_face(self, face):
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

