import copy
import os
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
import argparse
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
import cv2
import time
import shutil
from pathlib import Path
def parse_args():
    parser = argparse.ArgumentParser(description='OCR inference.')
    parser.add_argument('config', help='config log')
    parser.add_argument(
        '--weight', help='The checkpoint file to load from.')
    parser.add_argument(
        '--images', help='image dir')
    parser.add_argument(
        '--out_dir', help='out dir')
    args = parser.parse_args()
    return args

def main():
   args = parse_args()
   recog_config = args.config
   recog_ckpt = args.weight
   recog_model = init_detector(recog_config, recog_ckpt,device='cuda:0')
   if recog_model.cfg.data.test['type'] == 'ConcatDataset':
         recog_model.cfg.data.test.pipeline = \
            recog_model.cfg.data.test['datasets'][0].pipeline

   ROOT = args.out_dir
   if os.path.exists(ROOT):
         shutil.rmtree(ROOT)
   os.mkdir(ROOT)
   dir = Path(args.images)
   images = list(dir.glob('*'))
   for   image in images :
         name = os.path.basename(image)
         with open(os.path.join(args.out_dir,f'{name}.txt'),'w') as file :
            src_img = mmcv.imread(image)
            h, w = src_img.shape[:2]
            with open(os.path.join('results/detection/ensemble',f'{name}.txt'),'r') as f :
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

                  box_img = crop_img(src_img, box,long_edge_pad_ratio=0.02,short_edge_pad_ratio=0.02)
                  height, width =box_img.shape[:2]
                  ocr_results =  model_inference(recog_model, box_img)

                  save = [[x,y] for (x,y) in zip(points_x,points_y)]
                  score = round(ocr_results['score'],4)
                  text = ocr_results['text'].replace('<UKN>','').replace('α',' ')
                  # if len(text)==0:
                  #    continue
               
                  new_box = np.array(save).flatten().tolist()
                  text_score = text + '\t' + str(score)
                  # new_box.append(text_score)
                  file.write(','.join([str(i) for i in new_box]) + '\t' + text_score + '\n')      
if __name__ == '__main__':
   main()