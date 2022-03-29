
from mmocr.models.textdet.postprocess.wrapper import poly_nms
import cv2
import os
import glob
import numpy as np
import tqdm
from text_classification.net import Network
from text_classification.inference import text_classifier
from torchvision import transforms
import torch 
from mmocr.datasets.pipelines.crop import crop_img
import mmcv
import shutil
from PIL import Image
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
labels_to_index = {'text':0,'no_text':1}
index_to_label = {j:i for i,j in labels_to_index.items()}
transform = transforms.Compose([
            transforms.Resize((100,100)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
net = Network(num_class=2,is_train = False).to(device)
MODEL_SAVE_PATH = "/home/thorpham/Documents/mmocr/text_classification/models/weight-b3.pt"
net.load_state_dict(torch.load(MODEL_SAVE_PATH))
net.eval()



def get_box(path_file):
   with open(path_file,'r') as file:
      boxs = []
      data = file.readlines()
      for line in data:
         box = line.split(',')
         boxs.append(box)
   return boxs

db_640 = sorted(glob.glob('results/detection/640x640/*.txt'))
db_1280 = sorted(glob.glob('results/detection/1280x1280/*.txt'))
db_1400 = sorted(glob.glob('results/detection/1400x1400/*.txt'))
db_1600 = sorted(glob.glob('results/detection/1600x1600/*.txt'))
db_1800 = sorted(glob.glob('results/detection/1800x1800/*.txt'))
db_2200 = sorted(glob.glob('results/detection/2200x2200/*.txt'))
db_2600 = sorted(glob.glob('results/detection/2600x2600/*.txt'))
assert len(db_640) == len(db_1280), 'file not match'
assert len(db_1280) == len(db_1600), 'file not match'
assert len(db_1600) == len(db_2200), 'file not match'
assert len(db_2200) == len(db_2600), 'file not match'

root_image = 'TestB1/'
print('Run ensemble')
t1 = time.time()
if os.path.exists('results/detection/ensemble'):
        shutil.rmtree('results/detection/ensemble')
os.mkdir('results/detection/ensemble')
for (p0,p1,p2,p3,p4,p5,p6) in tqdm.tqdm(zip(db_640,db_1280,db_1400,db_1600,db_1800,db_2200,db_2600)):
   name_image = os.path.basename(p1).replace('.txt','')
   src_img = mmcv.imread(os.path.join(root_image,name_image))
   h, w = src_img.shape[:2]
   name_save = os.path.basename(p1)
   with open(os.path.join('results/detection/ensemble',name_save),'w') as file:
        boxs0 = get_box(p0)
        boxs1 = get_box(p1)
        boxs2 = get_box(p2)
        boxs3 = get_box(p3)
        boxs4 = get_box(p4)
        boxs5 = get_box(p5)
        boxs6 = get_box(p6)
        box_ensemble = poly_nms(boxs6 + boxs5 +boxs4 + boxs3 + boxs2 + boxs1 + boxs0 ,threshold=0.1)
        for boxs in box_ensemble:
            bboxs = boxs[:8]
            score =  boxs[8:9][0]
            if float(score) < 0.9 :
            
                points_x = [min(max(int(x), 0), w) for x in bboxs[0:8:2]]
                points_y = [min(max(int(y), 0), h) for y in bboxs[1:9:2]]

                min_x = min(points_x)
                min_y = min(points_y)
                max_x = max(points_x)
                max_y = max(points_y)
                box = [
                    min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
                ]

                box_img = crop_img(src_img, box,long_edge_pad_ratio=0.01,short_edge_pad_ratio=0.01)
                img = Image.fromarray(box_img).convert('RGB')
                pred = text_classifier(img,net,transform,index_to_label,device)
                if pred == 'no_text' :
                    cv2.imwrite(os.path.join('text_classification/no-text',f'{score}_{name_image}'),box_img)
                    continue
            file.write(','.join([i.strip() for i in boxs]) + '\n')
t2 = time.time()
print('Total time ensemble',t2-t1)
