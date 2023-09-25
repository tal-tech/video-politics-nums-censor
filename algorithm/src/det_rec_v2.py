# -*- encoding: utf-8 -*-
'''
@File    :   political_det.py
@Time    :   2020/05/07 15:29:51
@Author  :   houqi 
@Version :   1.0
@Contact :   houqi@100tal.com
@License :   (C)Copyright 2020
@Desc    :   None
'''

# here put the import lib
import torch
from face_detector.retinanet_onnx import Retinanet as retinanet
from face_recognize.pb_inference import face_recogize_model
import numpy as np
import src.config_retina as config


class Pipline():
    def __init__(self,det_path,recog_path):
        print('[Det] Creating networks and loading parameters...')
        self.face_detector = retinanet(det_path, config)

        print('[Rec] Loding face recognize models...')
        self.face_recogize = face_recogize_model(recog_path)

        print('[Rec] FACE_THRESHOLD:',config.FACE_THRESHOLD)
        

    def init_gallery(self,gallery):
        """
            init face gallery
            [
                { "img_group_id": 123,
                  "features": [ { "img_id": 1, "img_feature": "xxxxxxx" },
                                { "img_id": 2, "img_feature": "xxxxxxx" } ]
                }
            ]
        """
        self.gallery = gallery
        sub_gallery , _ = self.get_specify_gallery(lib_id=0)
        self.sub_gallery = sub_gallery
        
        for i in range(len(sub_gallery['features'])):
            if i == 0:
                self.sub_matrix = sub_gallery["features"][i]["img_feature"]
            else:
                self.sub_matrix = np.concatenate([self.sub_matrix, sub_gallery["features"][i]["img_feature"]])
                
        self.sub_matrix = self.sub_matrix.reshape(332, 1024)
        
        self.denum2 = np.linalg.norm(self.sub_matrix, axis=1)
        self.sub_matrix = torch.from_numpy(self.sub_matrix).cuda()
        self.denum2 = torch.from_numpy(self.denum2).cuda()
        self.denum2 = self.denum2.unsqueeze(1)


    def exract_features(self, img, flag):
        """[summary]
            特征提取接口，调用人脸检测模型和人脸识别模型提取人脸特征向量
        Args:
            img ([array]): [特征提取的图片，形状为（h, w, 3)，通道顺序为BGR]
            flag ([string]): [flag=“0”是初始化，则不需要人脸检测，直接提取人脸特征向量；
                              flag=“1” 检测人物中心人脸，并提取特征，用于增加人脸；
                              flag=“2”是正常的人脸检测+特征提取。]
        """

        if isinstance(img,type(None)) or img.shape[0]<=1 or img.shape[1]<=1:
            print("img is invalid")
            return None

        if flag ==0:
            # 直接提取人脸特征
            face_embedding = self.face_recogize.generate_single_embedding(img)

        elif flag ==1 :
            # 检测中心人脸
            crop_face = self.face_detector.detect_face_for_gallery(img)

            if not isinstance(crop_face,type(None)):
                face_embedding = self.face_recogize.generate_single_embedding(crop_face)
            else:
                print('img have no face, please check img...')
                return None

        else:
            # 检测所有人脸
            bboxes, _, _ = self.face_detector.detect_face(img)

            if len(bboxes) == 0:
                print('no det face...')
                return None

            batch_imgs = []
            # get all boxes
            for crop_face,bbox,_ in bboxes:
                batch_imgs.append(crop_face)
            
            # get batch embeddings （n,1024）
            face_embedding = self.face_recogize.generate_batch_embeddings(batch_imgs)
            assert face_embedding.shape[0] ==len(batch_imgs),'feat shape is not equal to batch_size'


        return face_embedding


    def add_id(self, feature, img_id,id_name, lib_id=0):
        """[summary]
         向人脸gallery库中添加新的人脸，一张图片最好包含一个正脸，如果包含多个人脸，根据人脸大小和图像中心选择中心人脸。
        Args:
            feature ([array]): [需要增加人脸的特征向量，维度为1024]
            lib_id ([int]): [图库id，取决于业务方自己维护的gallery 图库id]
            img_id ([int]): [需要gallery的图片id，唯一标识符]

        Returns:
            BOOL]: 添加成功返回True，否则返回False
        """
        
        # get specify gallery
        sub_gallery, g_index = self.get_specify_gallery(lib_id)
        # get similarity matrix
        sim_mat = np.zeros((1, len(sub_gallery["features"])))
        for j in range(len(sub_gallery["features"])):
            tmp_dist = self.cal_cos_dist(feature,np.array(sub_gallery["features"][j]["img_feature"]))
            sim_mat[0,j] = tmp_dist

        # 对比选取相似度最高的人脸
        # 默认argmax 返回平铺的最大索引
        tmp_x, tmp_y = np.unravel_index(sim_mat.argmax(), sim_mat.shape)

        if sim_mat[tmp_x][tmp_y] > config.ID_THRESHOLD:
            print('add_id with gallery id {} have similarity {} greater than threshold {}'.format(sub_gallery["features"][tmp_y]["img_id"],sim_mat[tmp_x][tmp_y], config.ID_THRESHOLD))
            return False

        else:
            # update gallery and id_list 
            tmp_dict = {"img_id": img_id, "img_feature": feature}
            self.gallery[g_index]["features"].append(tmp_dict)
            # print(self.gallery[g_index]["features"])
            return True


    def del_id(self,img_id_list,lib_id=0):
        """[summary]
            从gallery 库中删除指定id
        Arguments:
            img_id_list {[list]} -- [需要删除人物对应的img_id,如"温家宝"，则对应人物所有的img_id 均需要给出]
            lib_id ([int]): [图库id，取决于业务方自己维护的gallery 图库id]

        Returns:
            [bool] -- [删除成功返回True,否则返回False]
        """

        # get specify gallery
        sub_gallery, g_index = self.get_specify_gallery(lib_id)

        del_flag = False
        for tmp_index ,tmp_id in enumerate(self.gallery[g_index]["features"]):
            if tmp_id["img_id"] == img_id_list:
                self.gallery[g_index]["features"].pop(tmp_index)
                del_flag = True
                # print(self.gallery[g_index]["features"])
        return del_flag

    def query(self,feature,lib_id=0,face_threshold=config.FACE_THRESHOLD):
        """[summary]
            检测图片中是否包含涉政人物，如果有涉政人物，返回大于阈值的id和置信度，否则返回None
        Args:
            feature ([array]): [img {[array]} -- [查询图片中人脸的特征向量]
            lib_id ([string]) 图库id，取决于业务方自己维护的gallery 图库id
            face_threshold ([float], optional): [置信度阈值]. Defaults to config.FACE_THRESHOLD.

        Returns:
            tmp_id_list [list]: [输入图片中包含的大于阈值的人物id]
            tmp_sim_list [list]: [输入图片中包含的大于阈值的id对应的置信度]
        """
        
        # matrix similarity in cpu
        # denum1 = np.linalg.norm(feature, axis=1)
        # sim_mat = np.dot(feature / np.expand_dims(denum1, axis=1), (self.sub_matrix / np.expand_dims(self.denum2, axis=1)).T)

        # matrix similarity in GPU
        denum1 = torch.from_numpy(np.linalg.norm(feature, axis=1)).cuda().unsqueeze(1)
        feature = torch.from_numpy(feature).cuda()
        sim_mat = torch.matmul(feature / denum1, (self.sub_matrix / self.denum2).T) 
        sim_mat = sim_mat.cpu().numpy()
        
        # 对比选取相似度最高的人脸
        # 默认argmax 返回平铺的最大索引
        tmp_x, tmp_y = np.unravel_index(sim_mat.argmax(), sim_mat.shape)
        # 返回多个人脸的匹配id
        if sim_mat[tmp_x][tmp_y] < face_threshold:
            return None,None
        else:
            tmp_id_list = []
            tmp_sim_list = []
            while (sim_mat[tmp_x][tmp_y] >= face_threshold):
                tmp_id_list.append(self.sub_gallery["features"][tmp_y]["img_id"])
                tmp_sim_list.append(sim_mat[tmp_x][tmp_y])

                # 当前图片和当前人的距离置为-100.
                sim_mat[tmp_x,:] = -100.0
                sim_mat[:,tmp_y] = -100.0
                tmp_x, tmp_y = np.unravel_index(sim_mat.argmax(), sim_mat.shape)

            return tmp_id_list, tmp_sim_list
            

    def get_specify_gallery(self,lib_id):

        # get specify gallery
        
        for index,tmp_gallery in enumerate(self.gallery):
            if tmp_gallery["img_group_id"] == lib_id:
                return tmp_gallery, index


    def cal_cos_dist(self,emb1,emb2):
        """[summary]
        calulate cosine distance between embeddings , range [-1,1], 1 description simliarity
        Arguments:
            emb1 {[type]} -- [description]
            emb2 {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        assert emb1.shape==emb2.shape ,"embedding shape must be equal"
        dot_result = np.sum(emb1*emb2)
        emb1_norm = np.sqrt(np.sum(np.square(emb1)))
        emb2_norm = np.sqrt(np.sum(np.square(emb2)))
        #print dot_result, emb1_norm, emb2_norm
        return float(dot_result/(emb1_norm*emb2_norm))


if __name__ == "__main__":
    # face det
    det_path = "/Users/houqi/code/political_det/face_onnx/faceDetector_600x600.onnx"
    # face recognize 
    recog_path = "./output_model/face_recognize_v1.9.10.1.pb"

    