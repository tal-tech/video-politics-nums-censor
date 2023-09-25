# -*- encoding: utf-8 -*-
'''
@File    :   pb_inference.py
@Time    :   2020/03/24 11:26:17
@Author  :   houqi 
@Version :   1.0
@Contact :   houqi@100tal.com
@License :   (C)Copyright 2020
@Desc    :   None
'''

# here put the import lib

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import cv2
import os
import numpy as np
import time
import math
import codecs
from face_recognize.rec_util import *


class face_recogize_model(object):
    def __init__(self, model_path):

        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
        config = tf.compat.v1.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        config.gpu_options.allow_growth = False

        self.g=tf.compat.v1.Graph()
        self.sess=tf.compat.v1.Session(graph=self.g,config=config)
        with self.g.as_default():

            output_graph_def = tf.compat.v1.GraphDef()
            with open(model_path, "rb") as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.compat.v1.import_graph_def(output_graph_def, name='')

            # self.inputs = self.sess.graph.get_tensor_by_name('images:0')
            self.inputs = self.sess.graph.get_tensor_by_name('image:0')
            self.outputs= self.sess.graph.get_tensor_by_name('MobileFaceNet/Logits/SpatialSqueeze:0')


            self.embedding_size = self.outputs.get_shape()[1]
            self.image_height = 112
            self.image_width = 112

        print('load face recognize model...')

    def generate_batch_embeddings(self,paths,batch_size=5):
        nrof_images = len(paths)
        emb_array = np.zeros((nrof_images, self.embedding_size))
        nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
        for i in range(nrof_batches):
            start_index = i*batch_size
            # print('handing {}/{}'.format(start_index,nrof_images))
            end_index = min((i+1)*batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = load_data(paths_batch,(self.image_height,self.image_width))
            feats = self.sess.run(self.outputs, feed_dict={self.inputs:images})
            feats = l2_normalize(feats)
            emb_array[start_index:end_index,:] = feats
        return emb_array

    def generate_single_embedding(self,image_path):
        emb_array = self.generate_batch_embeddings([image_path],batch_size=1) 
        return emb_array[0]


def test(file_path,result_path):

    class_files = os.listdir(file_path)
    print("[FaceNet]Loading trained model...")

    # pb_path = "./output_model/face_recognize_v1.9.10.pb"
    # pb_path = "./output_model/face_recognize_v1.9.10.1.pb"
    pb_path = '/workspace/houqi/face_project/political_det/massface/deploy/pretrain_v1.0.2/model.pb'
    encoder = face_recogize_model(pb_path)

    avg_time = 0.
    total_count = 0.
    for class_file in class_files:
        print('process class :{}'.format(class_file))
        # Path to gallery
        gallery_dir = file_path + class_file + '/gallery/'
        gallery = load_gallery(gallery_dir)

        # Path to test data
        test_data_dir = file_path + class_file + '/test/'
        test_data = load_test_data(test_data_dir)

        # init gallery
        gallery_embedding = []
        for g_img in gallery:
            g_rep = encoder.generate_single_embedding(g_img)
            gallery_embedding.append(g_rep)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            if not os.path.exists(result_path + class_file):
                os.makedirs(result_path + class_file)
        # Path to model
        fd = open(result_path + class_file+'/'+'face_id_result.txt','w')

        # test
        correct = [0 for i in range(len(gallery))]
        group_count = [0 for i in range(len(gallery))]
        count = 0
        for index,img_path in enumerate(test_data):
            st = time.time()
            img = cv2.imread(img_path[0])
            l = img_path[1]
            test_rep = encoder.generate_single_embedding(img)
            dist = []
            for rep in gallery_embedding:
                tmp_dist = cal_cos_dist(rep,test_rep)
                dist.append(tmp_dist)
            tmp_l = np.argmax(dist)
            tmp_2 = max(dist)
            end =time.time()
            if tmp_l == l:
                correct[l] += 1
            group_count[l] += 1
            count += 1
            avg_time+=end-st
            write_line = str(img_path[0])+ ','+ str(tmp_l) + ',' + str(tmp_2) + '\n'
            fd.write(write_line)
            print("img:{} class:{} dist:{}".format(img_path[0],tmp_l,tmp_2))
            print("{}/{} time:{}".format(index,len(test_data),end-st))
            print ("{0} / {1} acc: {2}".format(np.sum(correct),count,float(np.sum(correct))/count))

        total_count += count
        print ("Overall test accuracy: {0:.5f}    of {1} images".format(float(np.sum(correct))/count,count))

    print('avg time:{}'.format(avg_time/total_count))


def precision_and_recall(result_path ,thr_min=0 ,thr_max=100, interval=1):
    class_files = os.listdir(result_path)
    thr_list = []
    for thr in range(thr_min, thr_max, interval):
        thr_list.append(thr / 100.0)
        total_hit = 0
        total_mistake = 0
        total_count = 0
        for class_file in class_files:

            f = codecs.open(result_path+class_file+'/'+'face_id_result.txt','r')
            result = f.readlines()

            hit = mistake = count =thr_count= 0
            for r in result:
                v = r.split(',')
                top1 = v[1]

                if float(v[-1]) >= thr / 100.0:
                    thr_count+=1
                    if v[0].split('/')[-2]== top1:
                        hit += 1
                    else:
                        mistake += 1
                count += 1
            total_hit = total_hit + hit
            total_mistake = total_mistake + mistake
            total_count = total_count + count

        if total_hit + total_mistake != 0:
            total_precision = total_hit / float(total_hit + total_mistake)
        total_recall = total_hit / float(total_count)

        if thr%1==0:
            print ("~~~~~~~~~~  thr:{0:.2f} ~~~~~~~~~~~~~ ".format(thr/100.0))
            print('P:{} R:{} F1:{}'.format(total_precision, total_recall, (2*total_precision*total_recall)/(total_precision+total_recall)))


if __name__ == "__main__":
     # 长期班
    # file_path = '/workspace/houqi/face_project/face_recognize/face_recog_data/test_set/85class_798_gallery/'
    # result_path = './pb_test/798/'

    # 短期班
    file_path = '/workspace/houqi/face_project/face_recognize/face_recog_data/test_set/398_test/'
    result_path = './pb_test/398/'

    test(file_path,result_path)
    precision_and_recall(result_path)

