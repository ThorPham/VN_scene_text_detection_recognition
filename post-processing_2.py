import numpy as np
import re
import mmcv
import cv2
import glob
import os
import shutil
from merger_phone import filter_posible_phone
from rule_base import rule1, rule2, check_pattern


ROOT = 'results/recognition/predicted'
if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
os.mkdir(ROOT)

def get_box(path_file):
   with open(path_file,'r') as file:
      boxs = []
      data = file.readlines()
      for line in data:
         box = line.split(',')
         boxs.append(box)
   return boxs

path_txts = sorted(glob.glob('results/recognition/rule/*'))
for txt in path_txts:
   name = os.path.basename(txt)
   boxs = get_box(txt)
   # meger phone
   boxs = filter_posible_phone(boxs)
   # split words 
   with open(os.path.join(ROOT,name),'w') as file :
         for line in boxs :
            file.write(','.join([str(i).strip() for i in line]) + '\n')
