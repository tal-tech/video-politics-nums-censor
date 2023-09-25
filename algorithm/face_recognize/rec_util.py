# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2020/03/24 14:41:25
@Author  :   houqi 
@Version :   1.0
@Contact :   houqi@100tal.com
@License :   (C)Copyright 2020
@Desc    :   None
'''

# here put the import lib
import os
import numpy as np
import time
import math
import cv2
import os.path as osp

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret
  

def load_data(image_paths, src_size=None):
	nrof_samples = len(image_paths)
	images = np.zeros((nrof_samples, src_size[0], src_size[1], 3))
	for i in range(nrof_samples):
		if image_paths[i] is str:
			img = cv2.imread(image_paths[i])[:,:,::-1]
		else:	
			img = image_paths[i][:,:,::-1]

		if src_size is not None:
			img = cv2.resize(img,(src_size[0],src_size[1]))

		if img.ndim == 2:
			img = to_rgb(img)
		
		img = img - 127.5
		img = img / 128.
		images[i,:,:,:] = img

	return images

def l2_normalize(x):
    n,e = x.shape
    mean = np.mean(x,axis=1)
    mean = mean.reshape((n,1))
    mean = np.repeat(mean,e,axis=1)
    x -= mean
    norm = np.linalg.norm(x,axis=1)
    norm = norm.reshape((n,1))
    norm = np.repeat(norm,e,axis=1)
    y = np.multiply(x,1/norm)
    return y

def load_gallery(data_dir):
	gallery_list = os.listdir(data_dir)
	gallery_img = [s for s in gallery_list if (s.endswith('.png') or s.endswith('.jpg'))]

	gallery = [0 for i in range(len(gallery_img))]
	for img_path in gallery_img:
		#label = int(img_path.split('.')[0])-150
		label = int(img_path.split('.')[0])
		img = cv2.imread(os.path.join(data_dir,img_path))
		# print(len(gallery), label)
		# print(img_path)
		gallery[label] = img
	return gallery

def get_all_images(path):
    filelist = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(('.jpg','.png','.jpeg')):
                filelist.append(os.path.join(root, name))
    print('There are %d images' % (len(filelist)))
    return filelist

def load_multi_gallery(data_dir):

	gallery_img =get_all_images(data_dir)
	label_list  = []
	gallery_list =[]

	for img_path in gallery_img:
		label_name = osp.basename(osp.dirname(img_path))
		img = cv2.imread(img_path)
		label_list.append(label_name)
		gallery_list.append(img)
	return gallery_list,label_list

def load_test_data(test_data_dir):
	file_list = os.listdir(test_data_dir)
	person_list = [p for p in file_list if not os.path.isfile(os.path.join(test_data_dir,p))]
	#print person_list
	img_path_list = []
	for p in person_list:
		person_path = os.path.join(test_data_dir,p)
		tmp_path = os.listdir(person_path)
		#print tmp_path
		for img in tmp_path:
			if (img.endswith('.png') or img.endswith('.jpg')):
				# num
				# img_pair = (os.path.join(test_data_dir,p,img),int(p))
				# chinese
				img_pair = (os.path.join(test_data_dir,p,img),p)
				img_path_list.append(img_pair)
	return img_path_list

def cal_cos_dist(emb1,emb2):
	dot_result = np.sum(emb1*emb2)
	emb1_norm = np.sqrt(np.sum(np.square(emb1)))
	emb2_norm = np.sqrt(np.sum(np.square(emb2)))
	#print dot_result, emb1_norm, emb2_norm
	return float(dot_result/(emb1_norm*emb2_norm))