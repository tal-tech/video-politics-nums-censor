# -*- encoding: utf-8 -*-
'''
@File    :   interface.py
@Time    :   2020/05/08 14:29:20
@Author  :   houqi 
@Version :   1.0
@Contact :   houqi@100tal.com
@License :   (C)Copyright 2020
@Desc    :   None
'''

import __init__
from src.det_rec_v2 import Pipline
from src.utils import get_all_images
from src import config_retina as config
import os
import os.path as osp
import time
import cv2
from tqdm import tqdm
import logging


log = logging.getLogger()
log.setLevel(logging.INFO)
log_file = logging.FileHandler('log_retina.txt', mode='w')
log.addHandler(log_file)

class Interface():
    def __init__(self,gallery_dir):

        self.Pipline = Pipline(config.det_path,config.recog_path)
        
        #init gallery
        gallery_list = self.build_gallery(gallery_dir)
        self.Pipline.init_gallery(gallery_list)


    def build_gallery(self,gallery_dir):

        gallery_list = [{"img_group_id":0,"features":[]}]
        self.tmp_id_list = []
        img_id = -1

        dir_list = [os.path.join(gallery_dir, files) for files in os.listdir(gallery_dir)]
        for i, dir_path in enumerate(dir_list):
            id_name = osp.basename(dir_path)
            if not osp.isdir(dir_path): continue
            for img_name in os.listdir(dir_path):
                if not img_name.endswith((".jpg",".png",".jpeg")):
                    continue
                img_path = osp.join(dir_path,img_name)
                img = cv2.imread(img_path)
                feature = self.Pipline.exract_features(img,flag=0)
                img_id += 1
                self.tmp_id_list.append(id_name)
                gallery_list[0]["features"].append({"img_id": img_id, "img_feature": feature})

        return gallery_list


    def metric(self,image_paths):
        
        correct_num = 0
        error_num = 0
        det_error_num = 0
        count = 0 
        index = 0
        mean_time = 0
        for i, img_path in tqdm(enumerate(image_paths), total=len(image_paths)):
            # log.info(img_path)
            index +=1
            label_name = osp.basename(osp.dirname(img_path))
            
            t1 = time.time()
            img = cv2.imread(img_path)
            features = self.Pipline.exract_features(img,flag=2)
            
            if isinstance(features,type(None)):
                det_error_num +=1
                ids,sim = None,None
            else:
                tmp_ids, sim = self.Pipline.query(features,lib_id=0)
                if tmp_ids is not None:
                    ids = []
                    for img_id in tmp_ids:
                        ids.append(self.tmp_id_list[img_id])
                else:
                    ids = tmp_ids

            if ids is not None:
                if label_name in ids:
                    correct_num+=1
                else:
                    error_num+=1
                    log.info('ERROR: img_path:{}  have ids:{} sim:{}'.format(img_path,ids,sim))

            else:
                log.info('miss img_path:{} ids:{}'.format(img_path,ids))
            t2 = time.time()
            
            mean_time += t2 - t1
            count += 1
            # log.info("index:{}/{} time:{} img_path:{}  have ids:{} sim:{}".format(index,len(image_paths),st2-st1,img_path,ids,sim))
            precision = correct_num/(correct_num+error_num) if (correct_num+error_num) !=0  else 0
            recall = correct_num/count

            f1 =(2*precision*recall)/(precision+recall) if (precision+recall) !=0  else 0
            log.info("precision:{} recall:{} f1:{} mean_time:{}".format(precision,recall,f1,mean_time/count))


        precision = correct_num/(correct_num+error_num)
        recall = correct_num/count
        f1 =(2*precision*recall)/(precision+recall)
        print("det_error:{} ".format(det_error_num))
        print('correct_num:{} error_num:{} total_count:{}'.format(correct_num,error_num,count))
        print("final precision:{} recall:{} f1:{} time:{}".format(precision,recall,f1,mean_time/len(image_paths)))
    


if __name__ == "__main__":
    
    # gallery_dir = "/workspace/houqi/face_project/political_det/face_recog_data/test_set/total_test/test/multi_gallery"
    
    # v2 测试集
    img_dir = "/dataset/all_data_0/v1_0/face_project/political_det/face_recog_data/test_set/total_ori_img"
    # 竞品测试集
    # img_dir = "/workspace/houqi/face_project/political_det/face_recog_data/test_set/competitor_test/imgs"
    # 攻击测试集
    # img_dir = "/dataset/all_data_0/v1_0/face_project/political_det/face_recog_data/test_set/attack_data"
    # 线上测试集
    # img_dir = "/dataset/all_data_0/v1_0/face_project/political_det/prepare_datasets/online_politic_data"
    # 原始测试集
    # img_dir = "/dataset/all_data_0/v1_0/face_project/political_det/face_recog_data/test_set/total_test/test/imgs"


    # gallery_dir = "/root/ldy/political_det/data/multi_gallery"

    # img_dir = "/root/ldy/political_det/data/test_imgs/imgs"

    gallery_dir = '/dataset/all_data_0/v1_0/face_project/political_det/face_recog_data/test_set/total_test/test/multi_gallery'
    # img_dir = '/dataset/all_data_0/v1_0/face_project/political_det/face_recog_data/test_set/competitor_test/imgs'
    # img_dir = '/share/wpc/face_project/political_det/face_recog_data/test_set/attack_data'

    images_list = get_all_images(img_dir)
    interface = Interface(gallery_dir)

    interface.metric(images_list)


    # for thres in tqdm.tqdm(thres_list):
    #     print('='*20 + '>', thres)

    #     images_list = get_all_images(img_dir)
    #     interface = Interface(gallery_dir)

    #     interface.metric(images_list)

