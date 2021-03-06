_base_ = [
    '../../_base_/schedules/schedule_adam_600e.py',
    '../../_base_/runtime_10e.py'
]
checkpoint_config = dict(interval=1)
model = dict(
    type='PANet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='caffe'),
    neck=dict(type='FPEM_FFM', in_channels=[256, 512, 1024, 2048]),
    bbox_head=dict(
        type='PANHead',
        in_channels=[128, 128, 128, 128],
        out_channels=6,
        loss=dict(type='PANLoss', speedup_bbox_thr=32)),
    train_cfg=None,
    test_cfg=None)

dataset_type = 'IcdarDataset'
data_root = 'data/icdar2017/'
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
    dict(
        type='ScaleAspectJitter',
        img_scale=[(3000, 800)],
        ratio_range=(0.7, 1.3),
        aspect_ratio_range=(0.9, 1.1),
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='PANetTargets', shrink_ratio=(1.0, 0.5)),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='RandomRotateTextDet'),
    dict(
        type='RandomCropInstances',
        target_size=(800, 800),
        instance_key='gt_kernels'),
    dict(type='Pad', size_divisor=32),
    # for visualizing img and gts, pls set visualize = True
    dict(
        type='CustomFormatBundle',
        keys=['gt_kernels', 'gt_mask'],
        visualize=dict(flag=False, boundary_key='gt_kernels')),
    dict(type='Collect', keys=['img', 'gt_kernels', 'gt_mask'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# data = dict(
#     samples_per_gpu=4,
#     workers_per_gpu=4,
#     val_dataloader=dict(samples_per_gpu=1),
#     test_dataloader=dict(samples_per_gpu=1),
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + '/instances_training.json',
#         img_prefix=data_root + '/imgs',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + '/instances_val.json',
#         img_prefix=data_root + '/imgs',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + '/instances_val.json',
#         img_prefix=data_root + '/imgs',
#         pipeline=test_pipeline))
# evaluation = dict(interval=10, metric='hmean-iou')
dataset_type1 = 'IcdarDataset'
data_root1 = 'data/icdar2015'
train11=dict(
    type=dataset_type1,
    ann_file=data_root1 + '/instances_training.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root1 + '/imgs',
    pipeline=train_pipeline)

train12=dict(
    type=dataset_type1,
    ann_file=data_root1 + '/instances_test.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root1 + '/imgs',
    pipeline=train_pipeline)

# # ctw1500
# dataset_type2 = 'IcdarDataset'
# data_root2 = 'data/ctw1500/'
# train2=dict(
#     type=dataset_type2,
#     ann_file=data_root2 + '/instances_training.json',
#     # for debugging top k imgs
#     # select_first_k=200,
#     img_prefix=data_root2 + '/imgs',
#     pipeline=train_pipeline)
# test2=dict(
#     type=dataset_type2,
#     ann_file=data_root2 + '/instances_test.json',
#     # for debugging top k imgs
#     # select_first_k=200,
#     img_prefix=data_root2 + '/imgs',
#     pipeline=test_pipeline)

#testocr

dataset_type3 = 'IcdarDataset'
data_root3 = 'data/textocr/'
train31=dict(
    type=dataset_type3,
    ann_file=data_root3 + '/instances_training.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root3 + '/imgs',
    pipeline=train_pipeline)
train32=dict(
    type=dataset_type3,
    ann_file=data_root3 + '/instances_val.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root3 + '/imgs',
    pipeline=train_pipeline)

#totaltext
dataset_type4 = 'IcdarDataset'
data_root4 = 'data/totaltext/'
train41=dict(
    type=dataset_type4,
    ann_file=data_root4 + '/instances_training.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root4 + '/imgs',
    pipeline=train_pipeline)
train42=dict(
    type=dataset_type4,
    ann_file=data_root4 + '/instances_test.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root4 + '/imgs',
    pipeline=train_pipeline)
#mydata
dataset_type5 = 'IcdarDataset'
data_root5 = 'data/mydata/'
train51=dict(
    type=dataset_type5,
    ann_file=data_root5 + '/instances_training.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root5 + '/imgs',
    pipeline=train_pipeline)

train52=dict(
    type=dataset_type5,
    ann_file=data_root5 + '/instances_test.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root5 + '/imgs',
    pipeline=train_pipeline)
# test5=dict(
#     type=dataset_type5,
#     ann_file=data_root5 + '/instances_test.json',
#     # for debugging top k imgs
#     # select_first_k=200,
#     img_prefix=data_root5 + '/imgs',
#     pipeline=test_pipeline)
dataset_type = 'IcdarDataset'
data_root = 'data/mydata/'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=2),
    train=dict(
        type='UniformConcatDataset',
        datasets=[train11, train12,train31,train32,train41,train42,train51],
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/instances_test.json',
        img_prefix=data_root + '/imgs',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/instances_test.json',
        img_prefix=data_root + '/imgs',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='hmean-iou')
