import numpy as np
import re
import mmcv
import cv2
import glob
import os
import shutil
from merger_phone import filter_posible_phone
from rule_base import rule1, rule2, check_pattern



ROOT = 'results/recognition/rule'
if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
os.mkdir(ROOT)


paths = glob.glob('results/recognition/ensemble/*')
for path in paths :
   base = os.path.basename(path)
   with open(path,'r') as file :
         data = file.readlines()
         rule_1 = []
         for line in data :
            boxs =  line.split(',')[:8]
            ocr = line.split(',')[8:]
            text = ','.join([i for i in ocr]).strip()
            boxs.append(text)
            rule_1.append(boxs)
            # rule 1
            if check_pattern(text,patterns=['fax','đt','đc','đ/c']): 
                out_bboxs = rule1(bboxs=boxs,patterns=['fax','sđt','đt','đc','đ/c',])
                rule_1.remove(boxs)
                for bb in out_bboxs:
                  rule_1.append(bb)

         rule_2 = []
         for r in rule_1 : 
            boxs =  r[:8]
            text = r[8]
            # text = ','.join(i for i in ocr).strip()
            boxs.append(text)
            # rule 2
            rule_2.append(boxs)
            if check_pattern(text,patterns=[' ']): 
               out_bboxs = rule2(bboxs=boxs,patterns=' ')
               rule_2.remove(boxs)
               for bb in out_bboxs:
                  rule_2.append(bb)
         with open(os.path.join(ROOT,base),'w') as f:
            for line in rule_2 :
               f.write(','.join([str(int(i)) for i in line[:8]]) + ',' + line[8] + '\n')
