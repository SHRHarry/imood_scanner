import os
import time
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from threading import Thread
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from api.src.controller.base_controller import BaseThreadController
from api.src.FaceBoxes.FaceBoxes import FaceBoxes
from api.src.models.PosterV2_7cls import pyramid_trans_expr2, load_pretrained_weights, MobileFaceNet
from api.src.utils import create_empty_canvas, insert_va_value, vis_img, get_face_roi, reshape_transform, _cv2_to_base64

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
    
class ModelThreadController(BaseThreadController):
    def __init__(self):
        logger.info('ceate ModelThreadController instance')
        super().__init__()
        self.face_model = None
        self.model = None
        self.is_success = False  
        
        self.pred_img = None
        self.attn_map = None
        self.vis_size = 700
        self.va_empty_fig = create_empty_canvas(self.vis_size)
        self.va_fig = self.va_empty_fig.copy()
        self.valence = None
        self.arousal = None
        self.expression = None
        self.duration = None
    
    def clean_va_fig(self):
        return {'va_fig': "data:image/png;base64,"+_cv2_to_base64(create_empty_canvas(self.vis_size)).decode('utf-8')}
    
    def upgate_va_fig(self):
        json_response = {'va_fig': None}
        
        if self.va_fig != None:
            json_response["va_fig"] = self.va_fig
        
        return json_response
    
    def init_model(self):
        if not self.is_running():
            self.mutex.acquire()
            try:
                self.process = ThreadWithReturnValue(target=self._load_model,)
                self.process.start()
                    
            except Exception as err:
                logger.error(f"[ERROR] | init_model | {err}")
                
            finally:
                self.mutex.release()
                
            return True
        return False
    
    def start_get_face_roi(self, img):
        if not self.is_running():
            self.mutex.acquire()
            try:
                self.process = ThreadWithReturnValue(target=get_face_roi,
                                                      args=(self.face_model, img,))
                self.process.start()
                rst = self.process.join()
                
            except Exception as err:         
                logger.error(f"[ERROR] | start_get_face_roi | {err}")

            finally:            
                self.mutex.release()
            
            return rst
        return None
    
    def start_predict(self, img, img_ori, dets):
        if not self.is_running():
            self.mutex.acquire()
            try:
                self.process = ThreadWithReturnValue(target=self._predict,
                                                     args=(img, img_ori, dets,))
                self.process.start()
                rst = self.process.join()
                
            except Exception as err:         
                logger.error(f"[ERROR] | start_predict | {err}")
                return None

            finally:            
                self.mutex.release()
            return rst
        
        return None
    
    def start_get_attn_map(self, img):
        if not self.is_running():
            self.mutex.acquire()
            try:
                self.process = ThreadWithReturnValue(target=self._get_attn_map,
                                                     args=(img,))
                self.process.start()
                rst = self.process.join()
                
            except Exception as err:         
                logger.error(f"[ERROR] | start_get_attn_map | {err}")
                return None

            finally:            
                self.mutex.release()
            return rst
        return None
    
    def _load_model(self,):
        try:
            self.face_model = FaceBoxes(timer_flag=False)
            self.model = pyramid_trans_expr2(img_size=300, num_classes=6)
            checkpoint = torch.load("./api/src/checkpoint/[08-30]-[00-10]-model_best.pth")
            self.model = load_pretrained_weights(self.model, checkpoint)
    
            self.model.eval()
            logger.info(f'self.face_model = {self.face_model}')
        except Exception as err:
            logger.error(f"[ERROR] | _load_model | {err}")
            self.face_model = None
            self.model = None
            
    def _predict(self, img, img_ori, dets):
        try:
            start = time.time()
            img_tensor = self._img_to_norm_tensor(img)
            
            with torch.no_grad():
                output = self.model(img_tensor)
                # predicted_idx = output[:, :4].argmax()
                arousal = output[:,4]
                valence = output[:,5]
                va_percentages, class_list = self._va_to_expression(valence, arousal)
            
            end = time.time()
            self.valence = float(valence.detach().cpu().numpy()[0])
            self.arousal = float(arousal.detach().cpu().numpy()[0])
            self.expression = class_list[np.argmax(va_percentages)]
            self.duration = float(end - start)
            
            vis_text = f"VA Predicton: {self.expression} | "
            for idx, value in enumerate(va_percentages):
                vis_text += f"{class_list[idx]} {value*100:.2f}% | "
            
            self.pred_img = "data:image/png;base64,"+_cv2_to_base64(vis_img(img_ori, dets=dets, text=vis_text, size=self.vis_size)).decode('utf-8')
            self.va_fig = "data:image/png;base64,"+_cv2_to_base64(insert_va_value(self.va_empty_fig.copy(), self.valence, self.arousal, img_size=self.vis_size)).decode('utf-8')
            logger.info(f'vis_text = {vis_text}')
            return (self.valence, self.arousal, self.expression, self.duration, self.pred_img, self.va_fig)

        except Exception as err:
            logger.error(f"[ERROR] | _predict | {err}")
            self.face_model = None
            self.model = None
            self.pred_img = None
            self.va_fig = self.va_empty_fig.copy()
            return None
    
    def _va_to_expression(self, valence, arousal):
        # print(f"valence.shape = {valence.shape}, arousal.shape = {arousal.shape}")
        # class_list =   ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
        # valence_list = [        0,    0.95, -0.85,       0.20,  -0.20,     -0.95,   -0.60,      -0.77]
        # arousal_list = [        0,    0.17, -0.38,       0.85,   0.87,      0.19,    0.75,       0.45]
        class_list =   ["Neutral", "Happy", "Sad", "Anger"]
        valence_list = [        0,   0.575, -0.85,   -0.63]
        arousal_list = [        0,   0.510, -0.38,   0.565]
        origin_length = len(class_list)
        
        intensity = torch.sqrt((valence - valence_list[0]) ** 2 + (arousal - arousal_list[0]) ** 2).reshape(1, -1)
        if intensity >= 0.2:
            del class_list[0]
            del valence_list[0]
            del arousal_list[0]
        
        for i in range(len(class_list)):
            if i == 0:
                x = torch.sqrt((valence - valence_list[i]) ** 2 + (arousal - arousal_list[i]) ** 2).reshape(1, -1)
            elif i > 0:
                x = torch.cat((x, torch.sqrt((valence - valence_list[i]) ** 2 + (arousal - arousal_list[i]) ** 2).reshape(1, -1)), 0)
        
        percentages = 1-self._softmax(x.detach().cpu().numpy().reshape(-1))
        if len(class_list) < origin_length:
            class_list.insert(0, "Neutral")
            percentages = np.append(0, percentages)
        
        return percentages, class_list
    
    def _softmax(self, x):
        x -= np.max(x, axis= 0, keepdims=True)
        f_x = np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

        return f_x
    
    def _get_attn_map(self, img):
        try:
            start = time.time()
            img_tensor = self._img_to_norm_tensor(img)
            
            target_layer = self.model.ffn2
            input_tensor = img_tensor
            cam = GradCAM(model=self.model, target_layer=target_layer, use_cuda=False, reshape_transform=reshape_transform)
            grayscale_cam = cam(input_tensor=input_tensor, target_category=None)
            visualization = show_cam_on_image(np.float32(img) / 255, grayscale_cam)
            end = time.time()
            self.duration = float(end - start)
            self.attn_map = "data:image/png;base64,"+_cv2_to_base64(vis_img(visualization, size=self.vis_size)).decode('utf-8')
            return (self.duration, self.attn_map)
        except Exception as err:
            logger.error(f"[ERROR] | _get_attn_map | {err}")
            self.face_model = None
            self.model = None
            return None
    
    def _img_to_norm_tensor(self, img):
        img_tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)
        
        img_tensor = torch.unsqueeze(img_tensor, dim=0)
        mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
        std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)
        img_tensor = img_tensor - mean.repeat(img_tensor.size(0), 1, img_tensor.size(2), img_tensor.size(3))
        img_tensor = img_tensor / std.repeat(img_tensor.size(0), 1, img_tensor.size(2), img_tensor.size(3))
        return img_tensor