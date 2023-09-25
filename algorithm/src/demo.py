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

# here put the import lib
import __init__
from src.det_rec_v2 import Pipline
from src.utils import get_all_images,get_acc
from src import config_retina as config
import os
import os.path as osp
import numpy as np
import cv2


class Interface():
    def __init__(self,gallery_dir):

        self.Pipline = Pipline(config.det_path,config.recog_path)
        
        #init gallery
        gallery_list = self.build_gallery(gallery_dir)
        self.Pipline.init_gallery(gallery_list)


    def build_gallery(self,gallery_dir):

        gallery_list = [{"img_group_id":0,"features":[]}]
        self.tmp_id_list = {}
        img_id = -1

        dir_list = [os.path.join(gallery_dir, files) for files in os.listdir(gallery_dir)]
        for dir_path in dir_list:
            id_name = osp.basename(dir_path)
            if not osp.isdir(dir_path): continue
            for img_name in os.listdir(dir_path):
                if not img_name.endswith((".jpg",".png",".jpeg")):
                    continue
                img_path = osp.join(dir_path,img_name)
                img = cv2.imread(img_path)
                feature = self.Pipline.exract_features(img,flag=0)
                img_id += 1
                self.tmp_id_list[img_id]=id_name
                gallery_list[0]["features"].append({"img_id": img_id, "img_feature": feature})

        return gallery_list


    def query(self, img, lib_id=0, face_threshold=config.FACE_THRESHOLD):

        result = {
            "code": "",
            "data": "",

        }
        try:
            features = self.Pipline.exract_features(img,flag=2)
            if features is None:
                result["code"] = 0
                result['data'] = []
                return result
            ids, sims = self.Pipline.query(features,lib_id=0)
            if ids is None:
                result["code"] = 0
                result['data'] = []
                return result
            img_result = []
            if not isinstance(img, type(None)):
                for id_, sim_ in zip(ids, sims):
                    id_ = self.tmp_id_list[id_]
                    img_result.append({id_:sim_})

            result["code"] = 0
            result["data"] = img_result

        except Exception as e:
            result["code"] = -1
            result["data"] = None

        finally:

            return result


    def add_id(self, img, img_id,id_name,lib_id=0):
      
        result ={
            "code": "",
            "data": "",

        }
        try:
            features = self.Pipline.exract_features(img,flag=1)
            flag = self.Pipline.add_id(features, img_id,id_name, lib_id)
            tmp_result = "failure"
            if flag:
                tmp_result = "success"
                self.tmp_id_list[img_id] =id_name

            result["code"] = 0
            result["data"] = tmp_result
            
        except:
            
            result["code"] = -1
            result["data"] = ""

        return result


    def del_id(self,img_id,lib_id=0):

        result ={
            "code": "",
            "data": "",

        }
        try:
            flag =  self.Pipline.del_id(img_id,lib_id=lib_id)
            tmp_result = "failure"
            if flag:
                tmp_result = "success"

                self.tmp_id_list.pop(img_id)

            result["code"] = 0
            result["data"] = tmp_result

        except: 

            result["code"] = -1
            result["data"] = ""

        return result
    
    def query_image_nums(self, img, lib_id=0, face_threshold=config.FACE_THRESHOLD):
        # 图片涉政人物数量检测
        res = self.query(img,0)
        if(res['code'] == 0):
            return len(res['data'])
        return 0
    
    def query_certify_video(self, img, lib_id=0, face_threshold=config.FACE_THRESHOLD):
        # 视频涉政人物检测可信度
        video_path = "input.mp4"
        image_dir = "images"
        cmd = "ffmpeg -i %s -vf fps=1 %s/out%%d.jpg" % (video_path,image_dir)
        os.system(cmd)

        files = os.listdir(image_dir)
        ans = 0
        for file in files:
            tmp1 = self.query(file, 0)
            if(tmp1['code'] == 0):
                tmp2 = tmp2['data']
                for k, v in tmp2:
                    ans += v
        
        return float(ans) / float(len(files))

    def query_video(self, img, lib_id=0, face_threshold=config.FACE_THRESHOLD):
        # 视频涉政人物检测
        video_path = "input.mp4"
        image_dir = "images"
        cmd = "ffmpeg -i %s -vf fps=1 %s/out%%d.jpg" % (video_path,image_dir)
        os.system(cmd)

        files = os.listdir(image_dir)
        ans = {}
        for file in files:
            tmp1 = self.query(file, 0)
            if(tmp1['code'] == 0):
                tmp2 = tmp2['data']
                for k, v in tmp2:
                    ans += k
        
        return ans


    def query_video_nums(self, img, lib_id=0, face_threshold=config.FACE_THRESHOLD):
        # 视频涉政人物数量检测
        return len(self.query_video(img,0))
    

if __name__ == "__main__":
    
    gallery_dir = "../examples/multi_gallery"
    interface = Interface(gallery_dir)

    img_path = '../examples/imgs/习近平/18870.jpg'
    img = cv2.imread(img_path)
    print(interface.query(img,0))
    print(interface.query_certify_video(img,0))

    
    img_path = '../examples/00001.jpg'
    
    img = cv2.imread(img_path)
    print(interface.add_id(img, id_name='习大大', img_id=9999, lib_id=0))
    print(interface.del_id(img_id=9999,lib_id=0))
