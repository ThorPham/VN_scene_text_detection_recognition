import copy
import os
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.image.misc import tensor2imgs
from mmcv.runner import load_checkpoint
from mmcv.utils.config import Config
import glob,os
from mmocr.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.core.visualize import det_recog_show_result
from mmocr.datasets.kie_dataset import KIEDataset
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.models import build_detector
from mmocr.utils.box_util import stitch_boxes_into_lines
from mmocr.utils.fileio import list_from_file
from mmocr.utils.model import revert_sync_batchnorm
from mmocr.core.visualize import draw_texts,draw_texts_by_pil
import tqdm 
import enchant
from supper_resolution.inference import predict, generator_init

import cv2
import time
t1 =  time.time()
model_scale = generator_init('supper_resolution/checkpoint/model_best_0.pth')

recog_config = '/home/thorpham/Documents/mmocr/RobustScanner/robustscanner_r31_academic.py'
recog_ckpt = '/home/thorpham/Documents/mmocr/RobustScanner/epoch_5.pth'
recog_model = init_detector(recog_config, recog_ckpt,device='cuda:0')
if recog_model.cfg.data.test['type'] == 'ConcatDataset':
      recog_model.cfg.data.test.pipeline = \
         recog_model.cfg.data.test['datasets'][0].pipeline

# load_dictionary 
dictionary = {}
with open('vn_dictionary.txt') as f :
   data = f.readlines()
   for line in data :
      word = line.strip().lower()
      dictionary[word] = word


def check_digit(text):
   condition = [i.isalpha() for i in text]
   if all(condition):
      return True
   return False
d = enchant.Dict("en_US")
images = sorted(glob.glob('/home/thorpham/Documents/TestB1/*'))
count = 0
for   image in images :

      name = os.path.basename(image)
      print('='*100)
      print(name)
      with open(os.path.join('outputs/final_round/predicted_new',f'{name}.txt'),'w') as file :
         src_img = mmcv.imread(image)
         h, w = src_img.shape[:2]
         with open(os.path.join('outputs/final_round/ensemble',f'{name}.txt'),'r') as f :
            data = f.readlines()
            for line in data:
               bboxs = line.split(',')
               
               points_x = [min(max(int(x), 0), w) for x in bboxs[0:8:2]]
               points_y = [min(max(int(y), 0), h) for y in bboxs[1:9:2]]

               min_x = min(points_x)
               min_y = min(points_y)
               max_x = max(points_x)
               max_y = max(points_y)
               box = [
                  min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
               ]

               box_img = crop_img(src_img, box,long_edge_pad_ratio=0.02,short_edge_pad_ratio=0.1)
               height, width =box_img.shape[:2]
               # if height < 32 :
               #    image_scale =  predict(box_img,model_scale)
               #    ocr_results =  model_inference(recog_model, image_scale)
               #    print(ocr_results)
               #    mmcv.imshow(image_scale,'image_scale')
               #    mmcv.imshow(box_img,'box_img')
               # else:
               ocr_results =  model_inference(recog_model, box_img)






               save = [[x,y] for (x,y) in zip(points_x,points_y)]
               score = round(ocr_results['score'],4)
               text = ocr_results['text'].replace('<UKN>','').replace('α',' ')
               # if len(text)==0:
               #    continue
               # if len(text)==1 and text=='-':
               #    continue
               # if score < 0.7 :
               #    if dictionary.get(text,None) == None :
               #       count +=1
               #       name = text + "_" +str(score) +'.jpg'
               #       cv2.imwrite(os.path.join('outputs/text_detection/failed',name),box_img)
               #       continue
               # name = text + "_" +str(score) +'.jpg'
               # if check_digit(text):
               #    if len(text) > 10 and score <0.99:
               #       if dictionary.get(text,None) == None and  d.check(text)==False:
               #          print('text 1',text)
               #          # cv2.imwrite(os.path.join('outputs/ensemble/failed',f'type_1_{name}'),box_img)
               #          continue

               #    if dictionary.get(text,None) == None and  d.check(text)==False and score<0.9:
               #       # cv2.imwrite(os.path.join('outputs/ensemble/failed',f'type_2_{name}'),box_img)
               #       print('text 2',text)
               # # if len(text) > 12:
               # #    continue

               # # if score<0.5:
               # #    continue
                     
               # if score<0.9:
               #     if dictionary.get(text,None) == None and  d.check(text)==False:
               #       #  cv2.imwrite(os.path.join('outputs/ensemble/failed',f'type_3_{name}'),box_img)
               #        print('text 3',text)
               #        continue
               
               #    print('text 3',text)
               # if box_img.shape[0] <16:
                 
               #    cv2.imwrite(os.path.join('outputs/ensemble/small',name),box_img)
                  # print('='*30)
                  # print('text ',text)
                  # print('score ',score)
                  # print('='*30)
                  # cv2.imshow('im',box_img)
                  # cv2.waitKey(0)
                  # cv2.destroyAllWindows()
               # cv2.imwrite(os.path.join('outputs/ensemble/outputs_ocr',name),box_img)
               new_box = np.array(save).flatten().tolist()
               new_box.append(text)
               file.write(','.join([str(i) for i in new_box]) + '\n')
            # if ocr_results['score'] < 0.7 :

      print('='*100)     
            # cv2.imwrite(os.path.join('outputs/text_detection/OCR',f'{ocr_results["text"]}_{score}.jpg'),box_img)
               # cv2.imshow('im',box_img)
               # cv2.waitKey(0)
               # cv2.destroyAllWindows()
t2=  time.time()
print('time ',t2-t1)