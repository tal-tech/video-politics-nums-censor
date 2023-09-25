# from numpy.core.fromnumeric import squeeze
import onnxruntime
import cv2
import torch
import numpy as np
# from layers.functions.prior_box import PriorBox
# from utils.nms.py_cpu_nms import py_cpu_nms
# from utils.box_utils import decode, decode_landm
# from data import cfg_rfb as cfg
import time
from tqdm import tqdm

torch.nn.ZeroPad2d


class ONNXModel():
    def __init__(self, onnx_path, im_height, im_width):
        """
        :param onnx_path:
        """
        self.im_height = im_height
        self.im_width = im_width
        
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        # print("input_name:{}".format(self.input_name))
        # print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed
    
    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        loc, conf, landms = self.onnx_session.run(self.output_name, input_feed=input_feed)
        
        return loc, conf, landms


def to_numpy(tensor):
    #print(tensor.device)
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# def main():
#     onnx_path = 'faceDetector.onnx'
#     img_path = './img/sample.jpg'
#     testset_folder = './data/politicface/val/images/'
#     # testset_list = args.dataset_folder[:-7] + "wider_val.txt"
#     testset_list = testset_folder[:-7] + "politic_val.txt"
    
#     with open(testset_list, 'r') as fr:
#         test_dataset = fr.read().split()
        
#         # testing begin
#     all_time = 0
#     num = 0
#     print(len(test_dataset))
#     for i, img_name in tqdm(enumerate(test_dataset)):
#         image_path = testset_folder + img_name
#         img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        
#         img = np.float32(img_raw)

#         # testing scale
#         target_size = 320
#         max_size = 320
#         im_shape = img.shape
#         im_size_min = np.min(im_shape[0:2])
#         im_size_max = np.max(im_shape[0:2])
#         resize = float(target_size) / float(im_size_min)
#         # prevent bigger axis from being more than max_size:
#         if np.round(resize * im_size_max) > max_size:
#             resize = float(max_size) / float(im_size_max)
#         if False:
#             resize = 1

#         if resize != 1:
#             img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
#         im_height, im_width, _ = img.shape
#         scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
#         img -= (104, 117, 123)
#         start_time = time.time()
#         model = ONNXModel(onnx_path, im_height, im_width)
#         end_time = time.time()
#         all_time += end_time - start_time
        
#         img_shape = img.shape
#         up = down = left = right = 0
#         if img_shape[0] < 320:      
#             up = down = (320 - img_shape[0]) // 2
#             if ((320 - img_shape[0]) % 2) != 0: 
#                 up += 1
#         if img_shape[1] < 320:
#             left = right = (320 - img_shape[1]) // 2
#             if ((320 - img_shape[1]) % 2) != 0: 
#                 left += 1
            
#         img_new = cv2.copyMakeBorder(img, up, down, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        
#         img = img_new.transpose(2, 0, 1)
#         img = np.expand_dims(img, axis=0)
#         # img = torch.from_numpy(img).unsqueeze(0)
#         # img = img.to(device)
#         loc, conf, landms = model.forward(img)
        
#         priorbox = PriorBox(cfg, image_size=(320, 320))
#         priors = priorbox.forward()
#         priors = priors
#         prior_data = priors.data
        
#         # decode bboxes
#         variances = cfg['variance']
#         priors = prior_data.cpu().detach().numpy()
#         loc = np.squeeze(loc.data)
#         boxes = np.concatenate((
#             priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
#             priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
#         boxes[:, :2] -= boxes[:, 2:] / 2
#         boxes[:, 2:] += boxes[:, :2]
        
#         # boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        
#         boxes = boxes * scale / resize
#         # boxes = boxes.cpu().numpy()
#         scores = np.squeeze(conf.data)[:, 1]
        
#         # decode landms
#         pre = np.squeeze(landms.data)
#         landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
#                         priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
#                         priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
#                         priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
#                         priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
#                         ), axis=1)
#         # landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
#         scale1 = [img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                                 img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                                 img.shape[3], img.shape[2]]
#         # scale1 = scale1.to(device)
#         landms = landms * scale1 / resize
#         # landms = landms.cpu().numpy()

#         # ignore low scores
#         confidence_threshold = 0.02
#         inds = np.where(scores > confidence_threshold)[0]
#         boxes = boxes[inds]
#         landms = landms[inds]
#         scores = scores[inds]

#         # keep top-K before NMS
#         order = scores.argsort()[::-1]
#         # order = scores.argsort()[::-1][:args.top_k]
#         boxes = boxes[order]
#         landms = landms[order]
#         scores = scores[order]

#         # do NMS
#         dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
#         nms_threshold = 0.4
#         keep = py_cpu_nms(dets, nms_threshold)
#         # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
#         dets = dets[keep, :]
#         landms = landms[keep]

#         # keep top-K faster NMS
#         # dets = dets[:args.keep_top_k, :]
#         # landms = landms[:args.keep_top_k, :]

#         dets = np.concatenate((dets, landms), axis=1)
#         num += 1
        
#     print('det_time{}', all_time / num)

# main()