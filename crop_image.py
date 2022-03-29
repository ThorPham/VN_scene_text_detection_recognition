import cv2
import numpy as np
import os
import glob
from mmocr.core.evaluation.utils import boundary_iou,poly_iou
from shapely.geometry import Polygon as plg
import random
import tqdm
images = glob.glob('/media/thorpham/PROJECT/challenge/mmocr/data/mydata/imgs/training/*')

root = '/media/thorpham/PROJECT/challenge/mmocr/data/mydata/annotations/training'
count = 0
for i in tqdm.tqdm(range(5000)):
    path = random.choice(images)
    im = cv2.imread(path)
    h, w = im.shape[:2]
    name = os.path.basename(path)   
    number = int(name.split('.')[0].replace('im',''))
    txt = os.path.join(root,'gt_' + str(number) + '.txt')
    list_polygon = []
    
    with open(txt,'r') as file :
       data = file.readlines()
       for line in data :
         bboxs = line.split(',')[:8]
         points_x = [min(max(int(x), 0), w) for x in bboxs[0:8:2]]
         points_y = [min(max(int(y), 0), h) for y in bboxs[1:9:2]]

         src_box = [[x,y] for x,y in zip(points_x,points_y)]

         list_polygon.append(src_box)
    while True :  
      crop_x = random.randint(0,w-64)
      crop_y = random.randint(0,h-64)
      box_crop = [
                     [crop_x, crop_y], [crop_x + 64, crop_y], [crop_x + 64, crop_y + 64] + [ crop_x, crop_y + 64]
                  ]

      areas =  [poly_iou(plg(box_crop),plg(box)) for box in list_polygon]
      if all(area< 0.0001 for area in areas ) :
         image_crop = im[crop_y:crop_y+64,crop_x:crop_x + 64]
         cv2.imwrite(os.path.join('/home/thorpham/Documents/mmocr/text_classification/datasets/no_text',f'gen_image_{i}.jpg'),image_crop)
         break

      # cv2.imshow('im',image_crop)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()
  