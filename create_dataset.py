import pandas as pd
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


recog_config = '/home/thorpham/Documents/mmocr/satrnet-full/satrn_vi.py'
recog_ckpt = '/home/thorpham/Documents/mmocr/satrnet-full/epoch_5.pth'
recog_model = init_detector(recog_config, recog_ckpt,device='cuda:0')
if recog_model.cfg.data.test['type'] == 'ConcatDataset':
      recog_model.cfg.data.test.pipeline = \
         recog_model.cfg.data.test['datasets'][0].pipeline

df = []
with open('data_ocr/mixture/my_ocr_data/train.txt','r') as file :
   data = file.readlines()
   for line in tqdm.tqdm(data):
      path = line.split(' ')[0]
      label = ' '.join(str(i) for i in line.split(' ')[1:])
      ocr_results =  model_inference(recog_model, os.path.join('data_ocr/mixture/my_ocr_data',path))
      text = ocr_results['text'].replace('<UKN>','').replace('α',' ')
      label = label.strip()
      # print(text)
      # print(label)
      df.append([label.lower(),text.lower()])

data_frame = pd.DataFrame(df,columns=['grouth_true','predict'])
data_frame.to_csv('data_ocr.csv',index=None)



'''
ố  -> õ
ố  -> ô
buyện -> huyện
â -> ậ 
ố  -> ó
d -> đ
viêc > việc
hượng > hướng
nhiét

'''