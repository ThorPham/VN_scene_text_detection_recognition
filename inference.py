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
from mmocr.core.visualize import draw_texts,draw_texts_by_pil
import time

def main(config,weight,image,output):

    cfg = Config.fromfile(config)
    model = init_detector(cfg, weight, device="cuda:0")
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][0].pipeline
    results = model_inference(model, image)
    out_file = os.path.join(output,os.path.basename(image))

    img = model.show_result(image, results, out_file=out_file, show=True)
    return img
if __name__ == "__main__":
    main()