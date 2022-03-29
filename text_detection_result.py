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
import time
import shutil
import warnings
warnings.filterwarnings("ignore")


# load detection model
def load_model_detection(scale,weight):
    print(f'Load model detection at scale {scale}')
    cfg = Config.fromfile(f'weights/text-detection/dbnet_r101_{scale}.py')
    checkpoint = weight
    model = init_detector(cfg, checkpoint, device="cuda:0")
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][0].pipeline
    return model
# inference


def run_detect(image_paths,model,save_dir,scale='640x640'):
    print(f'Start detect at scale  {scale}')
    images = glob.glob(image_paths + '/*')
    for image in tqdm.tqdm(images) :
        name = os.path.basename(image)
        with open(os.path.join(save_dir,scale,f'{name}.txt'),'w') as file :
            results = model_inference(model, image)
            src_img = mmcv.imread(image)
            h, w = src_img.shape[:2]
            for bbox in results['boundary_result']:
                boxs = bbox[:8]
                score = bbox[8:]
                if score[0] < 0.7:
                    continue
                
                points_x = [min(max(int(x), 0), w) for x in boxs[0:8:2]]
                points_y = [min(max(int(y), 0), h) for y in boxs[1:9:2]]
                save = [[x,y] for (x,y) in zip(points_x,points_y)]
                new_box = np.array(save).flatten().tolist()
                new_box.append(score[0])
                file.write(','.join([str(i) for i in new_box]) + '\n')

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    
    detect_scales = ['640x640','1280x1280','1400x1400','1600x1600','1800x1800','2200x2200','2600x2600']
    weight = '/home/thorpham/Documents/mmocr/db101-finetune-vn/epoch_200.pth'
    path_images = 'TestB1'
    root = 'results'
    if os.path.exists(os.path.join(root,'detection')):
        shutil.rmtree(os.path.join(root,'detection'))
    print('Create folder detection')
    os.mkdir(os.path.join(root,'detection'))
    t1 = time.time()
    for scale in detect_scales:
         print('='*100)
         os.mkdir(os.path.join(root,'detection',scale))
         model = load_model_detection(scale,weight)
         run_detect(path_images,model,save_dir=f'results/detection/',scale=scale)
         print('='*100)
    t2 = time.time()
    print('Total time  ',t2-t1)