import os
import glob
import numpy as np
import cv2


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
                thickness=1)
    return out_img

def get_box(path_file):
   with open(path_file,'r') as file:
      boxs = []
      data = file.readlines()
      for line in data:
         box = line.split(',')[:8]
         # box.append(1.0)
         boxs.append(box)
   return boxs

dictionary = []

with open('vn_dictionary.txt','r') as file:
   data = file.readlines()
   for word in data :
      dictionary.append(word.strip())

root_dir = '/home/thorpham/Documents/challenge/mmocr/data/mydata/imgs/test'
predicts =  glob.glob('data/mydata/annotations/test/*')
for path in predicts[100:] :
   basename = os.path.basename(path).replace('gt_','im').replace('txt','jpg')
   image = cv2.imread(os.path.join(root_dir,basename))
   print('image ',image.shape)
   with open(path,'r') as f :
      tmp = f.readlines()
   for line in tmp :
      output = line.strip().split(',')
      ocr_predict =  output[8:][0]
   boxs = get_box(path)
   img = draw_texts(image.copy(),boxs,color=(255,255,0))
   cv2.imshow('visualize',cv2.resize(img,(1000,600)))
   cv2.waitKey(0)
cv2.destroyAllWindows()