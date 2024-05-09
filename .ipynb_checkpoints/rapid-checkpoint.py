import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from dataset.coco_utils import get_coco_api_from_dataset
from dataset.coco_eval import CocoEvaluator
import utils
from PIL import Image ###add
import matplotlib.pyplot as plt ###add
from sklearn.preprocessing import StandardScaler ###add
from sklearn.cluster import DBSCAN ###add
import numpy as np
import pandas as pd
import coco_names
import random
import matplotlib
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import argparse
import json
import yaml
import pprint
from glob import glob
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from adversarial_robustness_toolbox.art.estimators.object_detection import PyTorchFasterRCNN
from adversarial_robustness_toolbox.art.attacks.evasion import RobustDPatch
from adversarial_robustness_toolbox.art.attacks.evasion import DPatch

import warnings ###add
warnings.filterwarnings(action='ignore')

from cuml import DBSCAN #add
from cuml.cluster import DBSCAN #add
import cudf # add
from sklearn.preprocessing import StandardScaler

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

######################### 이상치 찾기 #########################         
# def anomaly(window_size, stride, img_patch):
#     ker_var_list = []
    
#     for i in range(0,img_patch.shape[2], stride): 
#         for j in range(0,img_patch.shape[1], stride): 
#             ker = img_patch[:, j : min(img_patch.shape[1], j + window_size), 
#                                i : min(img_patch.shape[2], i + window_size)]
            
#             R = ker[0,:,:]
#             G = ker[1,:,:]
#             B = ker[2,:,:]
            
#             value = (np.var(R) + np.var(G) + np.var(B)) / 3 # pooled Variance
#             #value = (np.var(R)*np.var(G)*np.var(B))**(1/3) # Geometric Mean
#             ker_var_list.append(value)

#     ker_var_np = np.array(ker_var_list)
#     limit_var_lower = ker_var_np.mean() - (ker_var_np.std()*3)
#     limit_var_upper = ker_var_np.mean() + (ker_var_np.std()*3)
        
#     return limit_var_upper, limit_var_lower, ker_var_list
def anomaly(window_size, stride, img_array, device):
    img_patch = torch.from_numpy(img_array).to(device)

    R_img = img_patch[0]
    G_img = img_patch[1]
    B_img = img_patch[2]
    
    i_h, i_w = R_img.shape

    global R_pad
    global G_pad
    global B_pad
    
    remain1 = i_h % window_size 
    remain2 = i_w % window_size
    
    if remain1 == 0 and remain2 == 0:
            R_pad = torch.nn.functional.pad(R_img, (0, 0, 0, 0), mode='constant', value=0)
            G_pad = torch.nn.functional.pad(G_img, (0, 0, 0, 0), mode='constant', value=0)
            B_pad = torch.nn.functional.pad(B_img, (0, 0, 0, 0), mode='constant', value=0)
        
    elif remain1 == 0 and remain2 != 0:
            zero = window_size-remain2 
            R_pad = torch.nn.functional.pad(R_img, (0, zero, 0, 0), mode='constant', value=0)
            G_pad = torch.nn.functional.pad(G_img, (0, zero, 0, 0), mode='constant', value=0)
            B_pad = torch.nn.functional.pad(B_img, (0, zero, 0, 0), mode='constant', value=0)  
        
    elif remain1 != 0 and remain2 == 0:
            zero = window_size-remain1 
            R_pad = torch.nn.functional.pad(R_img, (0, 0, 0, zero), mode='constant', value=0)
            G_pad = torch.nn.functional.pad(G_img, (0, 0, 0, zero), mode='constant', value=0)
            B_pad = torch.nn.functional.pad(B_img, (0, 0, 0, zero), mode='constant', value=0)  

    elif remain1 != 0 and remain2 != 0:
            zero1 = window_size-remain1 
            zero2 = window_size-remain2
            R_pad = torch.nn.functional.pad(R_img, (0, zero2, 0, zero1), mode='constant', value=0)
            G_pad = torch.nn.functional.pad(G_img, (0, zero2, 0, zero1), mode='constant', value=0)
            B_pad = torch.nn.functional.pad(B_img, (0, zero2, 0, zero1), mode='constant', value=0) 
    
    
    #print(R_pad)
    #print(R_pad.shape)

    
    h,w = R_pad.shape
    strides = (stride*w,window_size,w,1)
    shapes = (int(h/window_size),int(w/window_size),window_size,window_size)
    
    #print(strides)
    #print(shapes)

    R_pathces = torch.as_strided(R_pad,
                              size=shapes,
                              stride=strides)
    G_pathces = torch.as_strided(G_pad,
                              size=shapes,
                              stride=strides)
    B_pathces = torch.as_strided(B_pad,
                              size=shapes,
                              stride=strides)

    #print(R_pathces)
    #print(R_pathces.shape)
    R_variance = torch.var(R_pathces*1.0, dim=(2,3))
    G_variance = torch.var(G_pathces*1.0, dim=(2,3))
    B_variance = torch.var(B_pathces*1.0, dim=(2,3))

    #print(R_variance)
    #print(R_variance.shape)
    ker_var_numpy = ((R_variance+ G_variance + B_variance)/3).cpu().numpy()
    #print(ker_var_numpy) 
    limit_var_upper = ker_var_numpy.mean() + 3 * ker_var_numpy.std()
    #print(ker_var_numpy.mean())
    limit_var_lower = ker_var_numpy.mean() - 3 * ker_var_numpy.std()
    ker_var_list = ((R_variance + G_variance + B_variance)/3).flatten().tolist()
    #print(ker_var_list)

    
    return limit_var_upper, limit_var_lower, ker_var_list
 
########################## 클러스터링 ##########################
def clustering(limit_var_upper, limit_var_lower, ker_var_list, window_size, stride, img_patch, device):
    x_coor = []
    y_coor = []

    idx = 0
    #print(len(ker_var_list))
    for i in range(0,img_patch.shape[1], stride): 
        for j in range(0, img_patch.shape[2], stride):
            # print('###################################')
            #print(img_patch.shape)
            # print(idx)   
            # print(i,j)
            if ker_var_list[idx] > limit_var_upper:
                x_coor.append(i)
                y_coor.append(j)
            if ker_var_list[idx] < limit_var_lower:
                x_coor.append(i)
                y_coor.append(j)
                
            idx += 1
         
    coor = pd.DataFrame({'x_coor':x_coor, 'y_coor':y_coor})
    mask_bboxes = []
    
    if not coor.empty:
        loc_data = StandardScaler().fit_transform(coor[['x_coor','y_coor']])
        loc_data = torch.from_numpy(loc_data).to(device)
        
        #clustering = DBSCAN(eps=0.1, min_samples=5).fit(loc_data) # multi
        clustering = DBSCAN(eps=0.07, min_samples=13).fit(loc_data) #singles
        
        coor['clust'] = clustering.labels_.get()
        
        max_idx = np.argmax(coor['clust'].value_counts())
        #print(coor['clust'].value_counts())

        keep = []
        for i in coor['clust'].value_counts().to_frame().index:
            if i != -1:
                keep.append(i)
            else:
                break

        if not coor['clust'].empty:
            for j in keep:
                coor_m = coor[coor["clust"] == j]
                #print(coor_m)

                for i in range(len(coor_m)):
                    img_patch[:, coor_m.iloc[i]["x_coor"]:coor_m.iloc[i]["x_coor"]+6, coor_m.iloc[i]["y_coor"]:coor_m.iloc[i]["y_coor"]+6] = 0
                ########### Masking half Exp ############
                #img_patch[:, int(x_min/2):int(x_max/2), y_min:y_max] = 0

            return img_patch
        else:
            return img_patch
    else:  
        return img_patch
##############################################################
#@torch.no_grad()
def evaluate(model, model2, data_loader, device, patchattack=False, defense=False):
    
    frcnn = PyTorchFasterRCNN(
                                model = model,
                                clip_values=(0, 1.0),
                                channels_first=False,
                                preprocessing_defences=None,
                                postprocessing_defences=None,
                                preprocessing=None,
                                attack_losses=(
                                    "loss_classifier",
                                    "loss_box_reg",
                                     "loss_objectness",
                                    "loss_rpn_box_reg",
                                ),
                                device_type=device)
    
    attack = RobustDPatch(
        frcnn,
        patch_shape=[120, 120, 3],
        patch_location=[0, 0],
        crop_range=[0, 0],
        brightness_range=[1.0, 1.0],
        rotation_weights=[1, 0, 0, 0],
        sample_size=1,
        learning_rate=0.1,
        max_iter=10,
        batch_size=1,
        targeted=False
    )
#     attack = DPatch(
#     frcnn,
#     patch_shape=(120, 120, 3),
#     learning_rate=0.01,
#     max_iter=10,
#     batch_size=1)
    
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
#     model.eval()
#     model.cuda()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    
    iou_lst = []
    ioa_lst = []
    speed = []
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        ############################################ ADD PATCH TO IMAGE ############################################
        if patchattack:
            img_np = []
            for img in image:
                img_np.append(img.numpy())
            img = np.array(img_np)
            #print(img)
            # print(img.shape)# (1, 3, 426, 640)
            # print(img.dtype)# float32
            img = img.transpose((0, 2, 3, 1)) ## (426, 640, 3)
            
            patch = attack.generate(img)
            image = attack.apply_patch(img, multi=False)
            # print('############################')
            # print(image[0].shape)# (1, 426, 640, 3)
            # print(image[0].dtype)# float32
            
            if defense:
                start = time.time() 
                img_array = image[0].copy()
                img_array = img_array.transpose((2,0,1))
                img_array = img_array.astype(np.float32)

                # from scipy.stats import skew, kurtosis
                # print(skew(ker_var_np, nan_policy='omit'))
                # if skew(ker_var_np, nan_policy='omit') < 4:
                #     img = clustering(limit_var_upper, limit_var_lower, ker_var_list, 3, 3, img_array)
                # else:
                #     img = img_array2
                
                limit_var_upper, limit_var_lower, ker_var_list  = anomaly(3, 3, img_array, device)
                img = clustering(limit_var_upper, limit_var_lower, ker_var_list, 3, 3, img_array, device) # (3, 640, 640)
                # print(img.shape) # (3, 426, 640)
                speed.append(time.time()-start)
                image = torch.from_numpy(img)
                image = image.unsqueeze(dim=0)
            
            else:
                image = image.transpose((0, 3, 1, 2)) 
                image = torch.from_numpy(image) # to tensor

            # print(image) 
            # print(image[0].shape) ###(n,3,640,586) # torch.Size([3, 426, 640])
            # print(image.dtype) # torch.float32
    

        ############################################ ADD PATCH TO IMAGE ############################################

            image = list(img.to(device) for img in image)
            # print('image', image)
            
        else:
            image = list(img.to(device) for img in image)
            # print(image)
            # print(image[0].shape) ### torch.Size([3, 426, 640])
            # print(image[0].dtype)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            
            ################################################### CHECK ##################################################
            
#         img_np = image[0].permute(1, 2, 0).cpu().numpy()

#         img_np[:, :, 0] = img_np[:, :, 0]*255.0 # B(0)
#         img_np[:, :, 1] = img_np[:, :, 1]*255.0 # G(1)
#         img_np[:, :, 2] = img_np[:, :, 2]*255.0 # R(2)
            
# #         print("Image shape:", img_np.shape)
# #         print("Image dtype:", img_np.dtype)
# #         print("Min pixel value:", np.min(img_np))
# #         print("Max pixel value:", np.max(img_np))
# #         print(img_np)

#         #img_pil = Image.fromarray(img_np.astype(np.uint8))
#         #plt.imshow(img_pil)
#         plt.imshow(img_np.astype(np.uint8))
#         plt.show()
        
            
    #         import seaborn as sns
    #         hm = sns.heatmap(img_np[:,:,0],cmap='coolwarm')
    #         hm.get_figure().savefig("/Data2/hm22/Faster-RCNN-with-torchvision-master/heatmap.png")
    
            
            ################################################### CHECK ##################################################
        with torch.no_grad():
            model2.eval()
            model2.cuda()
            # print('#############################################')
            # print(image)
            # print(image[0].shape)
            torch.cuda.synchronize()
            model_time = time.time()
            outputs = model2(image)

            ################################################### CHECK ##################################################
                
            # def random_color():
            #     b = random.randint(0,255)
            #     g = random.randint(0,255)
            #     r = random.randint(0,255)
            
            #     return (b,g,r)
                
            # boxes = outputs[0]['boxes'] # BOXES 4K COORDINATES (window 수 x anchor k수)
            # labels = outputs[0]['labels'] # RPN 통과 후, 어떤 물체인지 
            # scores = outputs[0]['scores']
            
            # tmp = image.copy()
            # tmp = tmp[0].permute(1, 2, 0).cpu().numpy()
            # #tmp = cv2.cvtColor(tmp,cv2.COLOR_RGB2BGR)
            
            # tmp[:, :, 0] = tmp[:, :, 0]*255.0 # B(0)
            # tmp[:, :, 1] = tmp[:, :, 1]*255.0 # G(1)
            # tmp[:, :, 2] = tmp[:, :, 2]*255.0 # R(2)
            
            # names = coco_names.names
            # for idx in range(boxes.shape[0]):
            #     if scores[idx] >= 0.5:
            #         bbox = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            #         x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            #         name = names.get(str(labels[idx].item()))
            #         display_str = "{}: {}%".format(name, int(100 * scores[idx]))
            #         # cv2.rectangle(img,(x1,y1),(x2,y2),colors[labels[idx].item()],thickness=2)
            #         # print(src_img.shape, x1,y1,x2,y2)
            #         x_min, y_min, x_max, y_max = map(int, bbox)
            #         cv2.rectangle(tmp, (x_min, y_min), (x_max, y_max), random_color(),thickness=2) 
            #         #cv2.rectangle(tmp,(int(x1),int(y1)),(int(x2),int(y2)),random_color(),thickness=-1)  
            #         text_size = cv2.getTextSize(display_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            #         #text_x = int(x1 + (x2 - x1) / 2 - text_size[0] / 2)
            #         #text_y = int(y2) + text_size[1] + 5#int(y1) - 5
            #         text_x = int(x2) + 5
            #         text_y = int(y1 + (y2 - y1) / 2 + text_size[1] / 2)
            #         cv2.putText(tmp, display_str, org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            #         fontScale=0.5, thickness=2, lineType=cv2.LINE_AA, color=(247, 81, 67))
            
            # plt.figure(figsize=(12, 8))
            # plt.axis("off")
            # #plt.imshow(image_with_boxes.astype(np.uint8), interpolation="nearest")
            # #src_img = cv2.cvtColor(src_img,cv2.COLOR_BGR2RGB)
            # plt.imshow(tmp.astype(np.uint8), interpolation="nearest")
            # plt.show()
                
            ################################################### CHECK ##################################################
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time
    
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
