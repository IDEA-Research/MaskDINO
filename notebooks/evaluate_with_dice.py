import sys
sys.path.append('/MaskDINO')
sys.path.append('/MaskDINO/detectron2/demo')

import argparse
from detectron2.engine import default_argument_parser
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import json
from collections import defaultdict
from skimage import draw
from tqdm import tqdm
from imantics import Polygons, Mask
from predictor import VisualizationDemo
import torch
from inference import Load_model

cfg = '/MaskDINO/runs/luggage_parts_1508_crop_imgenhance/config.yaml'
weight = '/MaskDINO/runs/luggage_parts_1508_crop_imgenhance/model_final.pth'

test_data = json.load(open('/MaskDINO/mnt_datasets/coco/luggage_parts/annotations/valid_1508.json'))
img_dir = '/MaskDINO/mnt_datasets/coco/luggage_parts/images/'
model = Load_model(cfg, weight)
conf = 0.3
# annos = json.load(open(ann_path))
img_to_anns = defaultdict(list)
for ann in test_data['annotations']:
    img_to_anns[ann['image_id']].append(ann)

# print(cfg.MODEL)
# inference = VisualizationDemo(cfg)


# from detectron2.utils.visualizer import GenericMask
# #convert mask
# def convert_mask(pred_masks):
#     mask = pred_masks.cpu().numpy()
#     ret=[]
#     for x in mask:
#         if isinstance(x, GenericMask):
#             ret.append(x)
#         else:
#             ret.append(GenericMask(x, vis_output.img.shape[0], vis_output.img.shape[1]))
#     return ret
def draw_binary_mask(seg, img_shape):
    seg = np.array(seg).reshape(-1,2)
    cvt_seg = seg.copy()
    cvt_seg[:,0] = seg[:,1]
    cvt_seg[:,1] = seg[:,0]
    mask = draw.polygon2mask(img_shape,cvt_seg)
    return mask
def intersection_check(original_image, mask1, mask2):

    intersection = (mask1*mask2).sum()
    iou = intersection/(mask1.sum()+mask2.sum())
    if iou > 0.2:
        # print('intersection', intersection)
        # print('iou', iou)
        return True
    return False
def draw_gt(img, anns):
    for ann in anns:
        seg = np.array(ann['segmentation']).reshape((-1,2)).astype(np.int32)
        # print('gt seg', seg.shape, seg)
        img = cv2.drawContours(img, [seg], -1,(255,255,0),10, cv2.LINE_AA)
    return img
# def draw_pred(img, seg):
#     polygons = Mask(seg).polygons()
#     polygons.points[0].shape
#     img = cv2.drawContours(img, polygons.points, -1, (0,255,255),3)
#     return img
def draw_pred(img, seg):
    seg = seg.reshape((-1,2)).astype(np.int32)
    # print('pred seg', seg.shape, seg)
    img = cv2.drawContours(img, [seg], -1,(255,0,255),3, cv2.LINE_AA)

    return img
    
dice_acc = defaultdict(list)

for img in tqdm(test_data['images']):
    # torch.cuda.empty_cache()
    img_path = os.path.join(img_dir, img['file_name'])
    image = cv2.imread(img_path)

    if image is not None:
        try:
            pred_boxes,pred_masks, pred_scores, pred_labels = model.detect(img_path, conf)
        except:
            # print('bug')
            continue
        else:
            # predictions, vis_output = inference.run_on_image(image)
            # preds = predictions['instances']
            # masks = convert_mask(preds.pred_masks)
            # labels = preds.pred_classes
            # scores = preds.scores
                
            gt_anns = img_to_anns[img['id']]
            for ann in gt_anns:
                # print('--------------------')
                cat_id = ann['category_id']
                # print('cat_id', cat_id)
        
                # draw_img = image.copy()
                # draw_img = draw_gt(draw_img,[ann])
                # plt.imshow(draw_img)
                gt_seg = ann['segmentation']
                gt_binary_mask = draw_binary_mask(gt_seg, image.shape[:2])
                # pred_masks = [m for m,l,s in zip(masks, labels, scores) if l.detach().cpu().tolist()==cat_id and s >= 0.3]
                # pred_scores = [s for m,l,s in zip(masks, labels, scores) if l.detach().cpu().tolist()==cat_id and s >= 0.3]
                # pred_binary_masks = [draw_binary_mask(np.array(pred.polygons), image.shape[:2]) for pred in pred_masks]
                pred_binary_masks = pred_masks
                # valid_polys = []
                # for pred in pred_masks:
                #     if len(pred.polygons)>1: #break 1 object thanh nhieu manh
                #         # print('len polygons',(len(pred.polygons)))
                #         # print('mask',pred._polygons)
                #         # for poly in pred.polygons:
                #         #     print('poly shape', poly.shape)
                #         #     draw_img = draw_pred(draw_img, poly)
                #         #     plt.imshow(draw_img)
                #         poly = np.array([max(pred.polygons, key = lambda x: x.shape[0])])
                #         valid_polys.append(poly)
                #         bi_mask = draw_binary_mask(poly,image.shape[:2])
                #     else:
                #         poly = np.array(pred.polygons)
                #         valid_polys.append(poly)
                #         bi_mask = draw_binary_mask(poly, image.shape[:2])
                #     pred_binary_masks.append(bi_mask)
                    
                # print('len pred',len(pred_binary_masks))
                if len(pred_binary_masks)>0:
                    for i,pred in enumerate(pred_binary_masks):
                        # draw_img = draw_pred(draw_img, valid_polys[i])
                        # plt.imshow(draw_img)
                        if intersection_check(image, pred, gt_binary_mask):
                            acc = (2*(gt_binary_mask*pred).sum() + 1)/(gt_binary_mask.sum()+pred.sum() + 1)
                            # print(acc)
                            dice_acc[cat_id].append(acc)
                else:
                    # acc = 0
                    dice_acc[cat_id].append(acc)

total_acc = []
for cls in dice_acc:
    total_acc.extend(dice_acc[cls])
print('dice_acc: ',sum(total_acc)/len(total_acc))