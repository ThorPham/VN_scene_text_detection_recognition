import os
import glob
import tqdm
import mmcv
import enchant
import cv2
import cv2
import numpy as np
import random
import tqdm
import os
import glob
# import shapely.geometry as sg
# from shapely import Polygon

def draw_texts(img, boxes=None, draw_box=True, on_ori_img=True,color= (0,0,255,0.8)):
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
    h, w = img.shape[:2]
   #  if boxes is None:
   #      boxes = [[0, 0, w, 0, w, h, 0, h]]

   #  if on_ori_img:
   #      out_img = img
   #  else:
    out_img = img
    for idx, boxs in enumerate(boxes):
      box = boxs[:8]
      text = boxs[8:][0]
      new_box = [[x, y] for x, y in zip(box[0::2], box[1::2])]
      min_x, max_x = min(box[0::2]), max(box[0::2])
      min_y, max_y = min(box[1::2]), max(box[1::2])
      Pts = np.array([new_box], np.int32)
      cv2.polylines(
            out_img, [Pts.reshape((-1, 1, 2))],
            True,
            color,
            thickness=1)
      # cv2.fillPoly(out_img, pts = [Pts], color =(255,random.randint(100,255),0,))

    return out_img
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
paths = glob.glob('/home/thorpham/Documents/mmocr/outputs/ensemble/submit/predicted/*')
count = 0
for path in tqdm.tqdm(paths) :
   base = os.path.basename(path).replace('.txt','')
   image = cv2.imread(os.path.join('/home/thorpham/Documents/TestA',base))
   with open(path,'r') as file :
      data = file.readlines()
      for line in data :
         boxs =  line.split(',')[:8]
         ocr = line.split(',')[8:]
         ocr = ','.join(i for i in ocr) 
         boxs.append(ocr)     # print(ocr)
         if dictionary.get(ocr.strip(),None)==None  :
            count+=1
            print(path)
            print(ocr)
            im1 = draw_texts(image.copy(),[boxs])
            cv2.imshow('un',cv2.resize(image,(1000,800)))
            cv2.imshow('img1',cv2.resize(im1,(1000,800)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
print(count)
