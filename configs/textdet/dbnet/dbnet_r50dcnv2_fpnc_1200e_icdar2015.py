_base_ = [
    '../../_base_/schedules/schedule_1200e.py', '../../_base_/runtime_10e.py'
]
# resume_from = '/home/thorpham/Documents/challenge/mmocr/dbnet/epoch_59.pth'
load_from = '/home/thorpham/Documents/challenge/mmocr/dbnet/epoch_59.pth'
total_epochs = 1200
checkpoint_config = dict(interval=1)
model = dict(
    type='DBNet',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPNC', in_channels=[256, 512, 1024, 2048], lateral_channels=256),
    bbox_head=dict(
        type='DBHead',
        text_repr_type='quad',
        in_channels=256,
        loss=dict(type='DBLoss', alpha=1.0, beta=10.0, bbce_loss=True)),
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
    dict(type='Normalize', **img_norm_cfg),
    # img aug
    dict(
        type='ImgAug',
        args=[['Fliplr', 0],
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
        img_scale=(4068, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(4068, 1024), keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
# total_epochs = 100
# icdar2015
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
        datasets=[train51],
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
evaluation = dict(interval=5, metric='hmean-iou')
