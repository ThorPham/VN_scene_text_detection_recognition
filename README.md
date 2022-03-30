# Vietnamses-science-text-recognition
## Requirement
- torch                     1.10.0                  
- torchvision               0.7.0   
- mmcv-full                 1.3.16
- mmdet                     2.18.1
- mmocr                     0.3.0
- opencv-python             4.5.4.60
- cudatoolkit               10.2.89
## TEST MODEL
- python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [ARGS]
- For eval detection --eval : 'hmean-iou', OCR --eval : 'acc'
## RUN PREDICT MODEL
* Inference.py (change file config path)
## Train model
- python tools/train.py ${CONFIG_FILE} [ARGS]
## Pretrained model
- Weight and some image visualization :[Link  Here](https://drive.google.com/drive/folders/11-HdUxw8BP_2e6rGrx7oot5LntfcAlzr?usp=sharing)