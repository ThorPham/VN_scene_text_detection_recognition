import numpy as np
import re
import mmcv
import cv2
import glob,os

def is_on_same_line(box_a, box_b, min_y_overlap_ratio=0.8):
    """Check if two boxes are on the same line by their y-axis coordinates.

    Two boxes are on the same line if they overlap vertically, and the length
    of the overlapping line segment is greater than min_y_overlap_ratio * the
    height of either of the boxes.

    Args:
        box_a (list), box_b (list): Two bounding boxes to be checked
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                                    allowed for boxes in the same line

    Returns:
        The bool flag indicating if they are on the same line
    """
    a_y_min = np.min(box_a[1::2])
    b_y_min = np.min(box_b[1::2])
    a_y_max = np.max(box_a[1::2])
    b_y_max = np.max(box_b[1::2])

    # Make sure that box a is always the box above another
    if a_y_min > b_y_min:
        a_y_min, b_y_min = b_y_min, a_y_min
        a_y_max, b_y_max = b_y_max, a_y_max

    if b_y_min <= a_y_max:
        if min_y_overlap_ratio is not None:
            sorted_y = sorted([b_y_min, b_y_max, a_y_max])
            overlap = sorted_y[1] - sorted_y[0]
            min_a_overlap = (a_y_max - a_y_min) * min_y_overlap_ratio
            min_b_overlap = (b_y_max - b_y_min) * min_y_overlap_ratio
            return overlap >= min_a_overlap or \
                overlap >= min_b_overlap
        else:
            return True
    return False


def stitch_boxes_into_lines(list_boxs, max_x_dist=10, min_y_overlap_ratio=0.6):
   """Stitch fragmented boxes of words into lines.

   Note: part of its logic is inspired by @Johndirr
   (https://github.com/faustomorales/keras-ocr/issues/22)

   Args:
      boxes (list): List of ocr results to be stitched
      max_x_dist (int): The maximum horizontal distance between the closest
                  edges of neighboring boxes in the same line
      min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                  allowed for any pairs of neighboring boxes in the same line

   Returns:
      merged_boxes(list[dict]): List of merged boxes and texts
   """


   # if len(boxes) <= 1:
   #    return boxes

   
   boxes = []
   for bb in list_boxs:
      box = [int(i) for i in bb[:8]]
      lb = ','.join(str(i) for i in bb[8:])
      boxes.append({'box' :box ,'text' :lb.strip()})
      # boxes['box'].append(bb[:8])
      # boxes['text'].append(bb[8:])
   merged_boxes = []

   # sort groups based on the x_min coordinate of boxes
   x_sorted_boxes = sorted(boxes, key=lambda x: np.min(x['box'][::2]))
   # store indexes of boxes which are already parts of other lines
   skip_idxs = set()

   i = 0
   # locate lines of boxes starting from the leftmost one
   count = 0
   for i in range(len(x_sorted_boxes)):
      if i in skip_idxs:
         continue
      # the rightmost box in the current line
      rightmost_box_idx = i
      line = [rightmost_box_idx]
      for j in range(i + 1, len(x_sorted_boxes)):
         if j in skip_idxs:
               continue
         if is_on_same_line(x_sorted_boxes[rightmost_box_idx]['box'],
                              x_sorted_boxes[j]['box'], min_y_overlap_ratio):
               line.append(j)
               skip_idxs.add(j)
               rightmost_box_idx = j
               # print('true')

      # split line into lines if the distance between two neighboring
      # sub-lines' is greater than max_x_dist
      lines = []
      line_idx = 0
      lines.append([line[0]])
      for k in range(1, len(line)):
         curr_box = x_sorted_boxes[line[k]]
         prev_box = x_sorted_boxes[line[k - 1]]
         dist = np.min(curr_box['box'][::2]) - np.max(prev_box['box'][::2])
         if dist > max_x_dist:
               line_idx += 1
               lines.append([])
         if len(lines[line_idx])>1:
               text = ''.join([x_sorted_boxes[idx]['text'].strip() for idx in lines[line_idx]]) + x_sorted_boxes[k]['text']
               # print('text',text)
               if sum(c.isdigit() for c in text.strip()) >11 :
                  line_idx += 1
                  lines.append([])
         lines[line_idx].append(line[k])
      # print('Lines',lines)

      # Get merged boxes
      for box_group in lines:
         if len(box_group) >= 2:
            if len(x_sorted_boxes[0]['text'].strip()) == 2:
               print('----------',x_sorted_boxes[0],x_sorted_boxes[1])
         merged_box = {}
         merged_box['text'] = ''.join(
               [x_sorted_boxes[idx]['text'].strip() for idx in box_group])
         x_min, y_min = float('inf'), float('inf')
         x_max, y_max = float('-inf'), float('-inf')
         for idx in box_group:
               x_max = max(np.max(x_sorted_boxes[idx]['box'][::2]), x_max)
               x_min = min(np.min(x_sorted_boxes[idx]['box'][::2]), x_min)
               y_max = max(np.max(x_sorted_boxes[idx]['box'][1::2]), y_max)
               y_min = min(np.min(x_sorted_boxes[idx]['box'][1::2]), y_min)
         merged_box['box'] = [
               x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max
         ]
         merged_boxes.append(merged_box)

   return merged_boxes

def check_character(text,list_check):
   for i in list_check:
      if i in text:
         return False
   return True

   # sort groups based on the x_min coordinate of boxes
   x_sorted_boxes = sorted(boxes, key=lambda x: np.min(x['box'][::2]))
def filter_posible_phone(bboxs):
   phone_boxs = []
   for box in bboxs :
      label = ','.join(str(i) for i in box[8:])

      temp = sum(c.isdigit() for c in label.strip())
      if temp >=2 and temp <=8 and check_character(text=label,list_check=[':','h','/','k','đ']):
         phone_boxs.append(box)
         # bboxs.remove(box)
      
   meger_boxs = stitch_boxes_into_lines(phone_boxs)
   for rm in phone_boxs :
      bboxs.remove(rm)
   for bb in meger_boxs:
      bbox = bb['box']
      lb = bb['text'].replace(' ','')
      bbox.append(lb)
      bboxs.append(bbox)
   return bboxs




def get_box(path_file):
   with open(path_file,'r') as file:
      boxs = []
      data = file.readlines()
      for line in data:
         box = line.split(',')
         boxs.append(box)
   return boxs



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
            box = box[:8]
            new_box = [[x, y] for x, y in zip(box[0::2], box[1::2])]
            Pts = np.array([new_box], np.int32)
            cv2.polylines(
                out_img, [Pts.reshape((-1, 1, 2))],
                True,
                color,
                thickness=1)
    return out_img
if __name__ == '__main__':
   # txts = glob.glob('/home/thorpham/Documents/challenge/mmocr/outputs/new_model/predicted/*')
   # for txt in txts:
   #    print('='* 30)
   #    name = os.path.basename(txt).replace('.txt','')
   #    boxs = get_box(txt)
   #    image = cv2.imread(f'/home/thorpham/Documents/challenge/TestA/{name}')
   #    img1 = draw_texts(image.copy(),boxs,color=(0,0,255,0.8))

   #    boxs = filter_posible_phone(boxs)
   #    img2 = draw_texts(image.copy(),boxs,color=(0,225,255,0.8))

   #    print('='* 30)
   #    mmcv.imshow(cv2.resize(np.concatenate([img1,img2],axis=0),(1200,1000)),'img')
      # mmcv.imshow(cv2.resize(img2,(1200,800)),'img2')
   root = '/home/thorpham/Documents/mmocr/outputs/ensemble/predicted'
   txts = glob.glob('/home/thorpham/Documents/mmocr/outputs/ensemble/submit/predicted/*')
   for txt in txts:
      name = os.path.basename(txt)
      boxs = get_box(txt)
      boxs = filter_posible_phone(boxs)
      with open(os.path.join(root,name),'w') as file :
         for line in boxs :
            file.write(','.join([str(i).strip() for i in line]) + '\n')