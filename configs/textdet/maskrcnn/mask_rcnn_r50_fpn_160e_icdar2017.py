_base_ = [
    '../../_base_/models/ocr_mask_rcnn_r50_fpn_ohem.py',
    '../../_base_/schedules/schedule_160e.py', '../../_base_/runtime_10e.py'
]

load_from = '/home/thorpham/Documents/challenge/mmocr/MaskRCNN_IC17/mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth'
# resume_from ='/home/thorpham/Documents/challenge/mmocr/MaskRCNN_IC17_continue/epoch_10.pth'
total_epochs = 30
checkpoint_config = dict(interval=1)
dataset_type = 'IcdarDataset'
data_root = 'data/mydata/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# img_norm_cfg = dict(mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='ScaleAspectJitter',
        img_scale=None,
        keep_ratio=False,
        resize_type='indep_sample_in_range',
        scale_range=(640, 1280)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='RandomCropInstances',
        target_size=(640, 640),
        mask_type='union_all',
        instance_key='gt_masks'),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='MultiScaleFlipAug',
        # resize the long size to 1600
        img_scale=(1600, 1600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # no flip
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


dataset_type = 'IcdarDataset'
data_root = 'data/mydata'
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=2),
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/instances_training.json',
        img_prefix=data_root + '/imgs',
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
#train11, train12,train31,train32,train41,train42,
