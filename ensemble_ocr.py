import cv2
import os
import glob
import numpy as np
import shutil
from collections import Counter
import json
import enchant


d = enchant.Dict("en_US")
dictionary = {}
with open('vn_dictionary.txt') as f :
   data = f.readlines()
   for line in data :
      word = line.strip().lower()
      dictionary[word] = word
# return box,text,score
def get_box(path_file):
   with open(path_file,'r') as file:
      boxs = []
      data = file.readlines()
      for line in data:
         box = line.split('\t')
         boxs.append(box)
   return sorted(boxs)

ROOT = 'results/recognition/ensemble'
if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
os.mkdir(ROOT)

root_NRTR = sorted(glob.glob('results/recognition/NRTR/*'))
root_RobustScanner = sorted(glob.glob('results/recognition/RobustScanner/*'))
root_satrnnet = sorted(glob.glob('results/recognition/satrnnet/*'))
assert len(root_NRTR) == len(root_RobustScanner), 'file not match'
assert len(root_RobustScanner) == len(root_satrnnet), 'file not match'

for (p1,p2,p3) in zip(root_satrnnet,root_RobustScanner,root_NRTR):
   name_file = os.path.basename(p1)
   ocr_1 = get_box(p1)
   ocr_2 = get_box(p2)
   ocr_3 = get_box(p3)
   with open(os.path.join(ROOT,name_file),'w') as file :
      for r1,r2,r3 in zip(ocr_1,ocr_2,ocr_3):
         b_1,text_1,score_1 = r1
         b_2,text_2,score_2 = r2
         b_3,text_3,score_3 = r3
         assert b_1 == b_2 , 'bbox not match'
         assert b_2 == b_3 , 'bbox not match'
         words= [text_1.strip(),text_2.strip(),text_3.strip()]
         scores = [score_1.strip(),score_2.strip(),score_3.strip()]
         words, scores = zip(*sorted(zip(words, scores),reverse=True))
         # print('='*50)
         # print(words)
         most_common,num_most_common = Counter(words).most_common(1)[0]
         if num_most_common > 1:
            ocr_result = most_common
         else :
            for i in range(len(words)) :
               if len(words[i]) == 0 :
                  continue
               if dictionary.get(words[i],None) != None and float(scores[i]) > 0.9:
                  ocr_result = words[i]
                  break
            else :
               max_value = max(scores)
               max_index = scores.index(max_value)
               ocr_result = words[max_index]
         # print('result',ocr_result)
         # print('='*50)
         if len(ocr_result) == 0:
            continue
         if ocr_result == '-' or ocr_result == '.':
            continue
         if ocr_result.startswith('-'):
            ocr_result = ocr_result[1:]

         save = b_1.split(',')
         save.append(ocr_result)
         file.write(','.join([str(i) for i in save]) + '\n')




