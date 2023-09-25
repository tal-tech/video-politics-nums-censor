import numpy as np
from config import cfg_rfb
import cv2
from prior_box import PriorBox
from py_cpu_nms import py_cpu_nms
from skimage import transform as trans
from onnx_inference import ONNXModel


class Retinanet():
    def __init__(self, det_path, config):
        self.det_path = det_path
        self.cfg = config
        self.height = self.cfg.height
        self.width = self.cfg.width
        self.model = ONNXModel(self.det_path, self.height, self.width)
        priorbox = PriorBox(cfg_rfb, image_size=(self.height, self.width))
        self.priors = priorbox.forward()
    
    def detect_face(self, img):
        target_size = self.cfg.long_side
        max_size = self.cfg.long_side
        im_shape = img.shape
        img = np.float32(img)
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        ori_img = img
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if self.cfg.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        img -= (104, 117, 123)
        
        img_shape = img.shape
        
        # padding image, input of onnx model
        up = down = left = right = 0
        if img_shape[0] < self.height:      
            up = down = (320 - img_shape[0]) // 2
            if ((320 - img_shape[0]) % 2) != 0: 
                up += 1
        if img_shape[1] < self.width:
            left = right = (320 - img_shape[1]) // 2
            if ((320 - img_shape[1]) % 2) != 0: 
                left += 1
        
        img_new = cv2.copyMakeBorder(img, up, down, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        scale = [img_new.shape[1], img_new.shape[0], img_new.shape[1], img_new.shape[0]]
        img = img_new.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        # forward pass
        loc, conf, landms = self.model.forward(img)  
        prior_data = self.priors.data
        
        # decode boxes
        variances = cfg_rfb['variance']
        priors = prior_data.cpu().detach().numpy()
        loc = np.squeeze(loc.data)
        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        
        # resize boxes
        boxes = boxes * scale 
        boxes[:, 0] -= left
        boxes[:, 2] -= left
        boxes[:, 1] -= up
        boxes[:, 3] -= up
        boxes /= resize
        # boxes = boxes.cpu().numpy()
        scores = np.squeeze(conf.data)[:, 1]
        
        # decode landms
        pre = np.squeeze(landms.data)
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), axis=1)
        scale1 = [img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]]

        # resize of landms
        landms = landms * scale1 
        for i in range(5):
            landms[:, i*2] = landms[:, i*2] - left
            landms[:, i*2+1] = landms[:, i*2+1] - up
        landms /= resize

        # ignore low scores
        inds = np.where(scores > self.cfg.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.cfg.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]
        
        # ingore small scores face
        landms = landms[dets[:, 4] > self.cfg.FACE_THRESHOLD]
        dets = dets[dets[:, 4] > self.cfg.FACE_THRESHOLD]

        dets = np.concatenate((dets, landms), axis=1)

        # pass each detected face
        face_list = []
        bboxes = []
        align_points = []
        for i in range(dets.shape[0]):
            height = int(dets[i][2]) - int(dets[i][0])
            width = int(dets[i][3]) - int(dets[i][1])
            # print height,width
            bx1 = np.maximum(int(dets[i][0] - height * self.cfg.CROP_RATIO), 0)
            by1 = np.maximum(int(dets[i][1] - width * self.cfg.CROP_RATIO), 0)
            bx2 = np.minimum(int(dets[i][2] + height * self.cfg.CROP_RATIO), 320)
            by2 = np.minimum(int(dets[i][3] + width * self.cfg.CROP_RATIO), 320)

            align_point = landms[i].copy()
            align_points.append(align_point)
            # 人脸对齐
            
            crop_face = self.align_face2(ori_img, align_point)

            bx1_ = np.maximum(int(dets[i][0]  - self.cfg.CROP_MARGIN), 0)
            by1_ = np.maximum(int(dets[i][1] - self.cfg.CROP_MARGIN), 0)
            bx2_ = np.minimum(int(dets[i][2] + self.cfg.CROP_MARGIN), 320)
            by2_ = np.minimum(int(dets[i][3] + self.cfg.CROP_MARGIN), 320)
            save_face = ori_img[by1_:by2_, bx1_:bx2_].copy()

            # 返回人脸图像+位置信息+resize前原图
            face_list.append((crop_face, [bx1,by1,bx2,by2], save_face))
            bboxes.append([bx1,by1,bx2,by2])
            
        return face_list, np.array(bboxes), np.array(align_points)
    
    def detect_face_for_gallery(self,image):
        img_matlab = image.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp
        #st = time.time()
        _, bounding_boxes, points= self.detect_face(img_matlab)

        num_faces = bounding_boxes.shape[0]
        # 判断检测到的人脸距离图像中心的偏移及人脸大小 —> 决定选取哪张脸；
        crop_face = None
        if num_faces>0:
            bbindex = 0
            if num_faces > 1:
                img_size = np.asarray(image.shape)[0:2]
                det = bounding_boxes[:,0:4]
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                img_center = img_size / 2
                offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                bbindex = np.argmax(bounding_box_size-offset_dist_squared*2.0)
            dst_points = points[bbindex]

            # crop_face = self.align_face(image, dst_points)
            crop_face = self.align_face2(image,dst_points)
        return crop_face

    
    def align_face2(self,image,points):
        """[summary]
        insightface 标准脸，points[5,2]
        标准脸：
               [[38.29459953 51.69630051]
                [73.53179932 51.50139999]
                [56.02519989 71.73660278]
                [41.54930115 92.3655014 ]
                [70.72990036 92.20410156]]
        Arguments:
            image {[type]} -- [description]
            points {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        M = None
        image_size=[112,112]
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041] ], dtype=np.float32 )
        # print(src.shape,points.shape)
        if image_size[1]==112:
            src[:,0] += 8.0
        dst = np.array([[points[2*j],points[2*j+1]] for j in range(5)]).astype(np.float32)

        tform = trans.SimilarityTransform()
        # import ipdb; ipdb.set_trace()
        tform.estimate(dst, src)
        M = tform.params[0:2,:]

        warped = cv2.warpAffine(image,M,(image_size[1],image_size[0]), borderValue = 0.0)

        return warped

