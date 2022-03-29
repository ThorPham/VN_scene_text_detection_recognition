import copy
import os
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
import cv2
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
import tqdm 
import time

def draw_texts(img, boxes=None, draw_box=True, on_ori_img=True,color= (0,0,255)):
    """Draw boxes and texts on empty img.

    Args:
        img (np.ndarray): The original image.
        texts (list[str]): Recognized texts.
        boxes (list[list[float]]): Detected bounding boxes.
        draw_box (bool): Whether draw box or not. If False, draw text only.
        on_ori_img (bool): If True, draw box and text on input image,
            else, on a new empty image.
    Return:
        out_img (np.ndarray): Visualized image.
    """
   #  color_list = gen_color()
    h, w = img.shape[:2]
    if boxes is None:
        boxes = [[0, 0, w, 0, w, h, 0, h]]

    if on_ori_img:
        out_img = img
    else:
        out_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for idx, box in enumerate(boxes):
        if draw_box:
            new_box = [[x, y] for x, y in zip(box[0::2], box[1::2])]
            Pts = np.array([new_box], np.int32)
            cv2.polylines(
                out_img, [Pts.reshape((-1, 1, 2))],
                True,
                color,
                thickness=2)
    return out_img


t1 = time.time()
# load detection model
cfg = Config.fromfile('/home/thorpham/Documents/mmocr/paper/OCR/NRTR/nrtr_r31_1by8_1by4_academic.py')
checkpoint = "/home/thorpham/Documents/mmocr/paper/OCR/NRTR/epoch_6.pth"
model = init_detector(cfg, checkpoint, device="cuda:0")
if model.cfg.data.test['type'] == 'ConcatDataset':
    model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][0].pipeline
# inference

images = glob.glob('/home/thorpham/Documents/mmocr/paper/data-test/OCR/*')
avg = []
for image in tqdm.tqdm(images) :
        im = cv2.imread(image)
        t1 = time.time()
        results = model_inference(model, image)
        t2 = time.time()
        print('time inference ',t2-t1)
        avg.append(t2-t1)
        # name = os.path.basename(image)
        name = results["text"]
        out_file = f"/home/thorpham/Documents/mmocr/paper/data-test/results-ocr/NRTR/{name}.jpg"
        cv2.imwrite(out_file,im)
        # out = [ i for i in  results["boundary_result"] if i[8] >0.7]
        # out = {"boundary_result" : out}
        # img = model.show_result(image, out, out_file=out_file, show=True)
        print(results["text"])
print('time-average ',np.mean(avg))
