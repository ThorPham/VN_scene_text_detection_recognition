_base_ = [
    '../../_base_/schedules/schedule_1200e.py', '../../_base_/runtime_10e.py'
]
resume_from = '/home/thorpham/Documents/mmocr/db-101-all/epoch_47.pth'
# load_from = '/home/thorpham/Documents/mmocr/db-101-all/epoch_47.pth'
total_epochs = 100
checkpoint_config = dict(interval=5)
model = dict(
    type='DBNet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPNC', in_channels=[256, 512, 1024, 2048], lateral_channels=256),
    bbox_head=dict(
        type='DBHead',
        text_repr_type='quad',
        in_channels=256,
        loss=dict(type='DBLoss', alpha=5.0, beta=10.0, bbce_loss=True)),
    train_cfg=None,
    test_cfg=None)


# img_norm_cfg = dict(
#    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# from official dbnet code
img_norm_cfg = dict(
    mean=[122.67891434, 116.66876762, 104.00698793],
    std=[58.395, 57.12, 57.375],
    to_rgb=False)
# for visualizing img, pls uncomment it.
# img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadTextAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='ColorJitter', brightness=32.0 / 255, saturation=0.5),
    dict(type='RandomScaling', size=800, scale=(0.5, 3)),
    dict(type='Normalize', **img_norm_cfg),
    # img aug
    dict(
        type='ImgAug',
        args=[['Fliplr', 0.4],
              dict(cls='Affine', rotate=[-10, 10]), ['Resize', [0.5, 3.0]]]),
    # random crop
    dict(type='EastRandomCrop', target_size=(640, 640)),
    dict(type='DBNetTargets', shrink_ratio=0.4),
    dict(type='Pad', size_divisor=32),
    # for visualizing img and gts, pls set visualize = True
    dict(
        type='CustomFormatBundle',
        keys=['gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'],
        visualize=dict(flag=False, boundary_key='gt_shrink')),
    dict(
        type='Collect',
        keys=['img', 'gt_shrink', 'gt_shrink_mask', 'gt_thr', 'gt_thr_mask'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 1600),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1600, 1600), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# total_epochs = 100
# # icdar2015
dataset_type = 'IcdarDataset'
data_root1 = 'data/icdar2015'
train1=dict(
    type=dataset_type,
    ann_file=data_root1 + '/instances_training.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root1 + '/imgs',
    pipeline=train_pipeline)

train2=dict(
    type=dataset_type,
    ann_file=data_root1 + '/instances_test.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root1 + '/imgs',
    pipeline=train_pipeline)

test1=dict(
    type=dataset_type,
    ann_file=data_root1 + '/instances_test.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root1 + '/imgs',
    pipeline=test_pipeline,
    test_mode=True)

#testocr


data_root3 = 'data/textocr'
train3=dict(
    type=dataset_type,
    ann_file=data_root3 + '/instances_training.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root3 + '/imgs',
    pipeline=train_pipeline)

train4=dict(
    type=dataset_type,
    ann_file=data_root3 + '/instances_val.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root3 + '/imgs',
    pipeline=train_pipeline)

test3=dict(
    type=dataset_type,
    ann_file=data_root3 + '/instances_val.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root3 + '/imgs',
    pipeline=test_pipeline)
#totaltext

data_root4 = 'data/totaltext'
train5=dict(
    type=dataset_type,
    ann_file=data_root4 + '/instances_training.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root4 + '/imgs',
    pipeline=train_pipeline)

train6=dict(
    type=dataset_type,
    ann_file=data_root4 + '/instances_test.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root4 + '/imgs',
    pipeline=train_pipeline)

test4=dict(
    type=dataset_type,
    ann_file=data_root4 + '/instances_test.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root4 + '/imgs',
    pipeline=test_pipeline)
#mydata
data_root5 = 'data/mydata'
train7=dict(
    type=dataset_type,
    ann_file=data_root5 + '/instances_training.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root5 + '/imgs',
    pipeline=train_pipeline)

train8=dict(
    type=dataset_type,
    ann_file=data_root5 + '/instances_test.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root5 + '/imgs',
    pipeline=train_pipeline)


test5=dict(
    type=dataset_type,
    ann_file=data_root5 + '/instances_test.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root5 + '/imgs',
    pipeline=test_pipeline)
    
data_root6 = 'data/data_testA'   
train9=dict(
    type=dataset_type,
    ann_file=data_root6 + '/instances_training.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root6 + '/imgs',
    pipeline=train_pipeline) 
test6=dict(
    type=dataset_type,
    ann_file=data_root6 + '/instances_training.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root6 + '/imgs',
    pipeline=test_pipeline)
data_root7 = 'data/data_testB'
train10=dict(
    type=dataset_type,
    ann_file=data_root7 + '/instances_training.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root7 + '/imgs',
    pipeline=train_pipeline) 
test7=dict(
    type=dataset_type,
    ann_file=data_root7 + '/instances_training.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root7 + '/imgs',
    pipeline=test_pipeline)


data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[train7,train8,train9,train10],
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=[test5,test6,test7],
        pipeline=test_pipeline),
    test=dict(
         type='UniformConcatDataset',
        datasets=[test5,test6,test7],
        pipeline=test_pipeline))
evaluation = dict(interval=5, metric='hmean-iou')