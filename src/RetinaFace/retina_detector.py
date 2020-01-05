import cv2
import sys
import numpy as np
import datetime
import os
import glob

from retinaface import RetinaFace


class RetinaDetector(object):
  def __init__(self):
    self.thresh = 0.8
    self.scales = [1024, 1980]

    # 0>=でGPU利用
    self.gpuid = -1
    self.model_path = os.path.join(os.path.dirname(__file__), 'model/retinaface-R50/R50')
    print("retina model path: ", self.model_path)
    self.detector = RetinaFace(self.model_path, 0, self.gpuid, 'net3')

  def detect(self, img):
    im_shape = img.shape
    target_size = self.scales[0]
    max_size = self.scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    #im_scale = 1.0
    #if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    processed_scales = [im_scale]
    flip = False

    faces, landmarks = self.detector.detect(img, self.thresh, scales=processed_scales, do_flip=flip)
    print("face shape in Retina: ", faces.shape)

    if faces is not None:
      print('find', faces.shape[0], 'faces num')
      return faces
    else:
      return None