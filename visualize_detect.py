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
from mmocr.datasets.pipelines.crop import crop_img, warp_img
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
def gen_color():
    """Generate BGR color schemes."""
    color_list = [(101, 67, 254), (154, 157, 252), (173, 205, 249),
                  (123, 151, 138), (187, 200, 178), (148, 137, 69),
                  (169, 200, 200), (155, 175, 131), (154, 194, 182),
                  (178, 190, 137), (140, 211, 222), (83, 156, 222)]
    return color_list

def draw_texts_by_pil(img, boxes=None, draw_box=True, on_ori_img=True):
    """Draw boxes and texts on empty image, especially for Chinese.

    Args:
        img (np.ndarray): The original image.
        texts (list[str]): Recognized texts.
        boxes (list[list[float]]): Detected bounding boxes.
        draw_box (bool): Whether draw box or not. If False, draw text only.
        on_ori_img (bool): If True, draw box and text on input image,
            else, on a new empty image.
    Return:
        out_img (np.ndarray): Visualized text image.
    """
    TRANSPARENCY = .75  # Degree of transparency, 0-100%
    OPACITY = int(255 * TRANSPARENCY)
    color_list = gen_color()
    h, w = img.shape[:2]
    # if boxes is None:
    #     boxes = [[0, 0, w, 0, w, h, 0, h]]
    # assert len(boxes) == len(texts)
    if on_ori_img:
        out_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        out_img = Image.new('RGB', (w, h), color=(255, 255, 255))
    out_draw = ImageDraw.Draw(out_img,'RGBA')
    # with open(save,'w') as file:
    for idx, boxs in enumerate(boxes):
        points_x = [min(max(int(x), 0), w) for x in boxs[0:8:2]]
        points_y = [min(max(int(y), 0), h) for y in boxs[1:9:2]]
        save = [(x,y) for (x,y) in zip(points_x,points_y)]
        color = tuple(list(color_list[idx % len(color_list)])[::-1])
        if draw_box:
            out_draw.polygon(save, fill=color + (OPACITY,), outline=(0,0,0))
        # font_path = os.path.join(dirname, 'font.TTF')
        font_path = os.path.join('font/tahomabd.ttf')
        # if not os.path.exists(font_path):
        #     url = ('http://download.openmmlab.com/mmocr/data/font.TTF')
        #     print(f'Downloading {url} ...')
        #     local_filename, _ = urllib.request.urlretrieve(url)
        #     shutil.move(local_filename, font_path)
        # fnt = ImageFont.truetype(font_path, font_size)
        # out_draw.text((min_x + 1, min_y + 1), text, font=fnt, fill=(0, 0, 0))

    del out_draw

    out_img = cv2.cvtColor(np.asarray(out_img), cv2.COLOR_RGB2BGR)

    return out_img
t1 = time.time()
# load detection model
cfg = Config.fromfile('db101-finetune-vn/dbnet_r101_vn.py')
checkpoint1 = "/home/thorpham/Documents/mmocr/db-101-all/epoch_47.pth"
checkpoint2 = "/home/thorpham/Documents/mmocr/db101-finetune-vn/epoch_200.pth"
model1 = init_detector(cfg, checkpoint1, device="cuda:0")
model2 = init_detector(cfg, checkpoint2, device="cuda:0")
if model1.cfg.data.test['type'] == 'ConcatDataset':
    model1.cfg.data.test.pipeline = model2.cfg.data.test['datasets'][0].pipeline
if model2.cfg.data.test['type'] == 'ConcatDataset':
    model2.cfg.data.test.pipeline = model2.cfg.data.test['datasets'][0].pipeline
# inference

images = glob.glob('/home/thorpham/Documents/AutoCrawler/download/bien_hieu/*')
count = 0 
for image in tqdm.tqdm(images) :
   name = os.path.basename(image)
    # save_txt = name.split('.')
   print(image)
   results1 = model_inference(model1, image)
   results2 = model_inference(model2, image)
   src_img = mmcv.imread(image)
   if len(results1['boundary_result']) == 0 :
      continue
   # img = model.show_result(image, results,show=True)
   h, w = src_img.shape[:2]
   out1 = draw_texts_by_pil(src_img,results1['boundary_result'])
   out2 = draw_texts_by_pil(src_img,results2['boundary_result'])
   cv2.imshow('image-1',out1)
   cv2.imshow('image-2',out2)

   cv2.waitKey(0)
cv2.destroyAllWindows()
t2 = time.time()
print('total time ',t2-t1)