import json
import cv2
import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_box(path_file):
   with open(path_file,'r') as file:
      boxs = []
      data = file.readlines()
      for line in data:
         box = line.split(',')
         boxs.append(box)
   return boxs

def gen_color():
    """Generate BGR color schemes."""
    color_list = [(101, 67, 254), (154, 157, 252), (173, 205, 249),
                  (123, 151, 138), (187, 200, 178), (148, 137, 69),
                  (169, 200, 200), (155, 175, 131), (154, 194, 182),
                  (178, 190, 137), (140, 211, 222), (83, 156, 222)]
    return color_list

def draw_texts_by_pil(img, boxes=None, draw_box=True, on_ori_img=True):
    """Draw boxes and texts on empty image, especially for Chinese.

    Args:
        img (np.ndarray): The original image.
        texts (list[str]): Recognized texts.
        boxes (list[list[float]]): Detected bounding boxes.
        draw_box (bool): Whether draw box or not. If False, draw text only.
        on_ori_img (bool): If True, draw box and text on input image,
            else, on a new empty image.
    Return:
        out_img (np.ndarray): Visualized text image.
    """
    TRANSPARENCY = .2 # Degree of transparency, 0-100%
    OPACITY = int(255 * TRANSPARENCY)
    color_list = gen_color()
    h, w = img.shape[:2]
    # if boxes is None:
    #     boxes = [[0, 0, w, 0, w, h, 0, h]]
    # assert len(boxes) == len(texts)
    out_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw_text = Image.new('RGB', (w, h), color=(255, 255, 255))
    out_draw = ImageDraw.Draw(out_img,'RGBA')
    draw = ImageDraw.Draw(draw_text)
    # with open(save,'w') as file:

    for idx, boxs in enumerate(boxes):
        box = list(map(int, boxs[:8]))
        # text = boxs[8:][0]
        text = ','.join(str(i) for i in boxs[8:])
        if len(text) < 1 :
            continue
        
        new_box = [[int(x), int(y)] for x, y in zip(box[0::2], box[1::2])]
        min_x, max_x = min(box[0::2]), max(box[0::2])
        min_y, max_y = min(box[1::2]), max(box[1::2])
      #   save = [(x,y) for (x,y) in zip(points_x,points_y)]
        points_x = [min(max(int(x), 0), w) for x in box[0:8:2]]
        points_y = [min(max(int(y), 0), h) for y in box[1:9:2]]
        save = [(x,y) for (x,y) in zip(points_x,points_y)]
        Pts = np.array([new_box], np.int32)
        color = tuple(list(color_list[idx % len(color_list)])[::-1])
        out_draw.polygon(save, fill=color + (OPACITY,), outline=(0,0,0))
        box_width = max(max_x - min_x, max_y - min_y)
        font_size = int( 2 *box_width / len(text))
        font_path = os.path.join('/home/thorpham/Documents/mmocr/font/tahomabd.ttf')
        fnt = ImageFont.truetype(font_path, font_size)
        draw.polygon(save, fill=color + (int(255 * 0.9),), outline=(0,0,0))
        draw.text((min_x + 1, min_y + 1), text, font=fnt, fill=(0, 0, 0))
       
      #   cv2.polylines(
      #       draw, [Pts.reshape((-1, 1, 2))],
      #       True,
      #       color,
      #       thickness=1)
    dst = Image.new('RGB', (out_img.width + draw_text.width , out_img.height))
    dst.paste(out_img, (0, 0))
    dst.paste(draw_text, (out_img.width,0))


    visualize = cv2.cvtColor(np.asarray(dst), cv2.COLOR_RGB2BGR)

    return visualize
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
    color_list = gen_color()
    h, w = img.shape[:2]
   #  if boxes is None:
   #      boxes = [[0, 0, w, 0, w, h, 0, h]]

   #  if on_ori_img:
   #      out_img = img
   #  else:
    out_img = img
    draw = np.ones((h, w, 3), dtype=np.uint8) * 255
    for idx, boxs in enumerate(boxes):
      box = boxs[:8]
      text = ','.join(str(i) for i in boxs[8:])
      new_box = [[x, y] for x, y in zip(box[0::2], box[1::2])]
      min_x, max_x = min(box[0::2]), max(box[0::2])
      min_y, max_y = min(box[1::2]), max(box[1::2])
      Pts = np.array([new_box], np.int32)
      cv2.polylines(
            out_img, [Pts.reshape((-1, 1, 2))],
            True,
            color,
            thickness=1)
      cv2.fillPoly(out_img, pts = [Pts], color =(255,255,0,0.1))
      box_width = max(max_x - min_x, max_y - min_y)
      font_size = int(0.9 * box_width / len(text))
      font_path = os.path.join('/home/thorpham/Documents/mmocr/font/tahomabd.ttf')
      fnt = ImageFont.truetype(font_path, font_size)
      draw.text((min_x + 1, min_y + 1), text, font=fnt, fill=(0, 0, 0))
      cv2.polylines(
            draw, [Pts.reshape((-1, 1, 2))],
            True,
            color,
            thickness=1)
    dst = Image.new('RGB', (out_img.width, out_img.height + draw.height))
    dst.paste(out_img, (0, 0))
    dst.paste(draw, (0, out_img.height))
    return dst
images = sorted(glob.glob('/home/thorpham/Documents/TestB1/*'))
root = 'results/recognition/predicted'
for path in images:
    im = cv2.imread(path)
    name = os.path.basename(path)
    # detect = cv2.imread(os.path.join('/home/thorpham/Documents/mmocr/outputs/ensemble/visualize',name))
    # number = int(name.split('.')[0].replace('im',''))
    txt = os.path.join(root,name +'.txt')
    boxs = get_box(txt)
    print(im.shape)
    print('='*30)
    print(path)
    print('='*30)
    img = draw_texts_by_pil(im.copy(),boxs)

    cv2.imwrite(os.path.join('results/visualize_full',name),img)
#     cv2.imshow('image',cv2.resize(img,(1500,800)))
#     cv2.imshow('im',cv2.resize(img,(1500,800)))
#     cv2.imshow('im1',cv2.resize(im,(1500,800)))


#     cv2.waitKey(0)
# cv2.destroyAllWindows()
