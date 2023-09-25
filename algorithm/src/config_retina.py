# -*- coding: utf-8 -*-
import os
import sys


# model path
base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# model path
# det_path = "../models/det_face_v1.0.0"
det_path = os.path.join(base_dir, 'models', 'rfb_onnx_models/faceDetector.onnx')
# recog_path = "../models/rec_face_model/model.pb"
recog_path = os.path.join(base_dir, 'models', 'rec_face_model/model.pb')

# # model path
# det_path = "../models/rfb_onnx_models/faceDetector.onnx"
# recog_path = "../models/rec_face_model/model.pb"


# face recognize 
FACE_THRESHOLD = 0.52
ID_THRESHOLD = 0.5
long_side = 320
confidence_threshold = 0.02
nms_threshold = 0.4
origin_size = False
CROP_MARGIN = 64
CROP_RATIO = 0.2
height = 320
width = 320