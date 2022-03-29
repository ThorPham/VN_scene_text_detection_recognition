_base_ = [
    '../../_base_/schedules/schedule_1200e.py',
    '../../_base_/default_runtime.py'
]
load_from = '/home/thorpham/Documents/challenge/mmocr/DRRG/drrg_r50_fpn_unet_1200e_ctw1500_20211022-fb30b001.pth'
total_epochs = 200
model = dict(
    type='DRRG',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPN_UNet', in_channels=[256, 512, 1024, 2048], out_channels=32),
    bbox_head=dict(
        type='DRRGHead',
        in_channels=32,
        text_region_thr=0.3,
        center_region_thr=0.4,
        link_thr=0.80,
        loss=dict(type='DRRGLoss')))
train_cfg = None
test_cfg = None

dataset_type = 'IcdarDataset'
data_root = 'data/mydata/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomScaling', size=800, scale=(0.75, 2.5)),
    dict(
        type='RandomCropFlip', crop_ratio=0.5, iter_num=1, min_area_ratio=0.2),
    dict(
        type='RandomCropPolyInstances',
        instance_key='gt_masks',
        crop_ratio=0.8,
        min_side_ratio=0.3),
    dict(
        type='RandomRotatePolyInstances',
        rotate_ratio=0.5,
        max_angle=60,
        pad_with_fixed_color=False),
    dict(type='SquareResizePad', target_size=800, pad_ratio=0.6),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='DRRGTargets'),
    dict(type='Pad', size_divisor=32),
    dict(
        type='CustomFormatBundle',
        keys=[
            'gt_text_mask', 'gt_center_region_mask', 'gt_mask',
            'gt_top_height_map', 'gt_bot_height_map', 'gt_sin_map',
            'gt_cos_map', 'gt_comp_attribs'
        ],
        visualize=dict(flag=False, boundary_key='gt_text_mask')),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_text_mask', 'gt_center_region_mask', 'gt_mask',
            'gt_top_height_map', 'gt_bot_height_map', 'gt_sin_map',
            'gt_cos_map', 'gt_comp_attribs'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 640),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1024, 640), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'TextDetDataset'
img_prefix1 = 'data/mydata/imgs'
train_anno_file1 = 'data/mydata/instances_training.txt'
train1 = dict(
    type=dataset_type,
    img_prefix=img_prefix1,
    ann_file=train_anno_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=4,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    pipeline=train_pipeline,
    test_mode=False)
#testocr
img_prefix2 = 'data/textocr/imgs'
train_anno_file2 = 'data/textocr/instances_training.txt'
train2 = dict(
    type=dataset_type,
    img_prefix=img_prefix2,
    ann_file=train_anno_file2,
    loader=dict(
        type='HardDiskLoader',
        repeat=4,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    pipeline=train_pipeline,
    test_mode=False)
#testocr

#totaltext
img_prefix3 = 'data/totaltext/imgs'
train_anno_file3 = 'data/totaltext/instances_training.txt'
train3 = dict(
    type=dataset_type,
    img_prefix=img_prefix3,
    ann_file=train_anno_file3,
    loader=dict(
        type='HardDiskLoader',
        repeat=4,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    pipeline=train_pipeline,
    test_mode=False)


img_prefix4 = 'data/icdar2015/imgs'
train_anno_file4 = 'data/icdar2015/instances_training.txt'
train4 = dict(
    type=dataset_type,
    img_prefix=img_prefix4,
    ann_file=train_anno_file4,
    loader=dict(
        type='HardDiskLoader',
        repeat=4,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    pipeline=train_pipeline,
    test_mode=False)

img_prefix1 = 'data/mydata/imgs'
test_anno_file1 = 'data/mydata/instances_test.txt'
test1 = dict(
    type=dataset_type,
    img_prefix=img_prefix1,
    ann_file=test_anno_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    pipeline=test_pipeline,
    test_mode=True)

img_prefix2 = 'data/textocr/imgs'
test_anno_file2 = 'data/textocr/instances_val.txt'
test2= dict(
    type=dataset_type,
    img_prefix=img_prefix2,
    ann_file=test_anno_file2,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    pipeline=test_pipeline,
    test_mode=True)


img_prefix3 = 'data/totaltext/imgs'
test_anno_file3 = 'data/totaltext/instances_test.txt'
test3= dict(
    type=dataset_type,
    img_prefix=img_prefix3,
    ann_file=test_anno_file3,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    pipeline=test_pipeline,
    test_mode=True)


img_prefix4 = 'data/icdar2015/imgs'
test_anno_file4 = 'data/icdar2015/instances_test.txt'
test4= dict(
    type=dataset_type,
    img_prefix=img_prefix4,
    ann_file=test_anno_file4,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineJsonParser',
            keys=['file_name', 'height', 'width', 'annotations'])),
    pipeline=test_pipeline,
    test_mode=True)

dataset_type = 'IcdarDataset'
data_root = 'data/mydata/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=2),
    train=dict(
        type='UniformConcatDataset',
        datasets=[train1],
        pipeline=train_pipeline),
    val=dict(
         type='UniformConcatDataset',
        datasets=[test1],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
       datasets=[test1],
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='hmean-iou')
