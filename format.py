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
         box = line.split(',')#[:8]
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
    TRANSPARENCY = .75  # Degree of transparency, 0-100%
    OPACITY = int(255 * TRANSPARENCY)
    color_list = gen_color()
    h, w = img.shape[:2]
    # if boxes is None:
    #     boxes = [[0, 0, w, 0, w, h, 0, h]]
    # assert len(boxes) == len(texts)
    if on_ori_img:
        out_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        out_img = Image.new('RGB', (w, h), color=(255, 255, 255))
    out_draw = ImageDraw.Draw(out_img,'RGBA')
    # with open(save,'w') as file:
    for idx, boxs in enumerate(boxes):
        points_x = [min(max(int(x), 0), w) for x in boxs[0:8:2]]
        points_y = [min(max(int(y), 0), h) for y in boxs[1:9:2]]
        save = [(x,y) for (x,y) in zip(points_x,points_y)]
        color = tuple(list(color_list[idx % len(color_list)])[::-1])
        if draw_box:
            out_draw.polygon(save, fill=color + (OPACITY,), outline=(0,0,0))
        # font_path = os.path.join(dirname, 'font.TTF')
        font_path = os.path.join('font/tahomabd.ttf')
        # if not os.path.exists(font_path):
        #     url = ('http://download.openmmlab.com/mmocr/data/font.TTF')
        #     print(f'Downloading {url} ...')
        #     local_filename, _ = urllib.request.urlretrieve(url)
        #     shutil.move(local_filename, font_path)
        # fnt = ImageFont.truetype(font_path, font_size)
        # out_draw.text((min_x + 1, min_y + 1), text, font=fnt, fill=(0, 0, 0))

    del out_draw

    out_img = cv2.cvtColor(np.asarray(out_img), cv2.COLOR_RGB2BGR)

    return out_img
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
            cv2.fillPoly(out_img, pts = [Pts], color =(255,255,0,0.1))
    return out_img
images = glob.glob('TestB1/*')
root = 'results/detection/ensemble'
# root0 = 'outputs/final_round/db-1000'
# root1 = 'outputs/final_round/db-1200'
# root2 = 'outputs/final_round/db-1400'
# root3 = 'outputs/final_round/db-1600'
# root4 = 'outputs/final_round/db-1800'
# root5 = 'outputs/final_round/db-2000'
for path in images:
    im = cv2.imread(path)
    name = os.path.basename(path)   
    # number = int(name.split('.')[0].replace('im',''))
    txt = os.path.join(root,name +'.txt')
    boxs = get_box(txt)

   
    print('='*30)
    print(path)
    print('='*30)
    img = draw_texts_by_pil(im.copy(),boxs)

    cv2.imwrite(os.path.join('results/visualize_detect',name),img)
    print('='*100)
    # print('ensemble ',boxs[-1])
    # print('db-460 ',boxs0[-1])
    # print('db-640 ',boxs1[-1])
    # print('db-1280 ',boxs2[-1])
    # print('db-1600 ',boxs3[-1])
    # print('db-2200 ',boxs4[-1])
    # print(img.shape)
    # print(name)
    # print('='*100)
    # cv2.imshow('image',img)
    # cv2.imshow('image-460',img0)
    # cv2.imshow('image-640',img1)
    # cv2.imshow('image-1280',img2)
    # cv2.imshow('image-1600',img3)
    # cv2.imshow('image-2200',img4)
    # cv2.imshow('image-2600',img5)

#     cv2.waitKey(0)
# cv2.destroyAllWindows()
