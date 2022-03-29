import cv2
import numpy as np
import random
import tqdm
import os
import glob
import shapely.geometry as sg
# import shapely.geometry as sg
# from shapely import Polygon
from mmocr.datasets.pipelines.box_utils import sort_vertex

def polygon_intersect_x(poly, x_val):
    """
    Find the intersection points of a vertical line at
    x=`x_val` with the Polygon `poly`.
    """
    
    if x_val < poly.bounds[0] or x_val > poly.bounds[2]:
        raise ValueError('`x_val` is outside the limits of the Polygon.')
    
    if x_val == poly.bounds[0]:
        x_val += 2
    if x_val == poly.bounds[2]:
        x_val -= 2
    if isinstance(poly, sg.Polygon):
        poly = poly.boundary
    vert_line = sg.LineString([[x_val, poly.bounds[1]],
                               [x_val, poly.bounds[3]]])
    pts = [pt.xy[1][0] for pt in poly.intersection(vert_line)]
    pts.sort()
    return pts



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
# split text by bounding box
# x1,y1,x2,y2,x3,y3,x4,y4,ocr
def split_text_by_bbox(bboxs,index=[3,6],space=False):
   # x1,y1,x2,y2,x3,y3,x4,y4 = 
   text = bboxs[-1]
#    assert 0 < index and index < len(text), 'index out of range'
   points_x = [int(x) for x in bboxs[:-1][0:8:2]]
   points_y = [int(y) for y in bboxs[:-1][1:9:2]]
   points_x, points_y = sort_vertex(points_x, points_y)
   x1,y1,x2,y2,x3,y3,x4,y4 = points_x[0],points_y[0],points_x[1],points_y[1],points_x[2],points_y[2],points_x[3],points_y[3]
   # poly = Polygon(bbox)
   length = len(text)
   if space:
    deta_x12 = abs((x2-x1)/(length - len(index)))
    deta_x34 = abs((x3-x4)/(length - len(index)))
   else:
    deta_x12 = abs((x2-x1)/length)
    deta_x34 = abs((x3-x4)/length)

   x1_start = x1
   x4_start = x4
   text_index_start = 0
   y_1_start =  y1
   y_2_start = y4
   result_boxs = []
   for k in range(len(index)) :
     x_1_end = int(deta_x12 * index[k] + x1)
     x_2_end = int(deta_x34 * index[k] + x4)
     # find y coordinates
     polygon = sg.Polygon([(x,y) for (x,y) in zip(points_x, points_y)])
     points_y1 = polygon_intersect_x(polygon,x_1_end)
     points_y2 = polygon_intersect_x(polygon,x_2_end)
     y_1_end = int(min(points_y1))
     y_2_end = int(max(points_y2))
     
     result_boxs.append([x1_start, y_1_start, x_1_end, y_1_end,x_2_end,y_2_end,x4_start,y_2_start,text[text_index_start:index[k]].strip()])
     text_index_start = index[k]
     x1_start = x_1_end
     x4_start = x_2_end
     y_1_start = y_1_end
     y_2_start = y_2_end
  
#    result_boxs = [x1, y1, x_1_new, y_1_new,x_2_new,y_2_new,x4,y4,text[:index].strip()]
   result_boxs.append([x1_start, y_1_start, x2, y2, x3, y3,x4_start,y_2_start,text[text_index_start:].strip()])
   return result_boxs

def check_pattern(text,patterns=['fax','đt','đc','đ/c','sđt']): 
    for p in patterns:
        if p in text :
            return True
    return False

def rule1(bboxs,patterns=['fax','sđt','đt','đc','đ/c']):
    text = bboxs[-1].strip() 
    for p in patterns :
        if  p == text.strip():
            bboxs[-1] =  bboxs[-1] 
            return [bboxs]    
        if p + ':' == text.strip():
            return [bboxs]
        if len(text)>3:
            pattern = p + ":"
            if pattern == text.strip():
                return [bboxs]
            if  pattern in text:
                spl =  [x for x in text.split(pattern)]
                if len(spl[0])==0:
                    index = [len(pattern)]
                else:
                    index = [len(spl[0]),len(spl[0]) + len(pattern)]
                boxs = split_text_by_bbox(bboxs=bboxs,index=index)
                return boxs
    return [bboxs]
    
def rule2(bboxs,patterns=' '):
    text = bboxs[-1].strip()
    spl =  text.split(patterns)
    # if all([i.isalpha() for i in spl]):
    start = 0
    index = []
    for i in range(len(spl)-1):
        index.append(start + len(spl[i]))
        start = len(spl[i]) + 1
    boxs = split_text_by_bbox(bboxs=bboxs,index=index,space=True)
    return boxs
    # return [bboxs]


if __name__ == "__main__":
    paths = glob.glob('/home/thorpham/Documents/vn-science-text-recognition/results/recognition/meger_phone/*')
    for path in tqdm.tqdm(paths) :
        base = os.path.basename(path).replace('.txt','')
        image = cv2.imread(os.path.join('/home/thorpham/Documents/TestB1',base))
        with open(path,'r') as file :
            data = file.readlines()
            for line in data :
                boxs =  line.split(',')[:8]
                ocr = line.split(',')[8:]
                ocr = ','.join(i for i in ocr).strip()
                boxs.append(ocr)
                # # # rule 1
                # if check_pattern(ocr,patterns=['fax','đt','đc','đ/c']): 
                #     out_bboxs = rule1(bboxs=boxs,patterns=['fax','sđt','đt','đc','đ/c',])
                # rule 2
                if check_pattern(ocr,patterns=[' ']): 
                    out_bboxs = rule2(bboxs=boxs,patterns=' ')
                    print('='*100)
                    print(path)
                    print('input',[boxs])
                    print('output',out_bboxs)
                    print('='*100)
                    im1 = draw_texts(image.copy(),[boxs])
                    im2 = draw_texts(image.copy(),out_bboxs)
                    cv2.imshow('img1',im1)
                    cv2.imshow('om',im2)
                    cv2.waitKey(0)
                cv2.destroyAllWindows




   
   # cv2.imshow('img2',im2)
   # bbs = [[1093,503],[1096,481],[1146,488],[1142,510]]
   # for i in range(len(bbs)):
   #    cv2.circle(image,bbs[i],3,(0,255,255))
 
