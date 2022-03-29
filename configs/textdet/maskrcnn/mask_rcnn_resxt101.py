_base_ = [
    '../../_base_/models/ocr_mask_rcnn_rxt101.py',
    '../../_base_/schedules/schedule_adam_600e.py', '../../_base_/runtime_10e.py'
]

# load_from = '/home/thorpham/Documents/challenge/mmocr/mask-rest101/latest.pth'
resume_from ='/home/thorpham/Documents/challenge/mmocr/MaskRCNN_full/epoch_1.pth'
total_epochs = 50
checkpoint_config = dict(interval=1)
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
        img_scale=(1280, 1280),
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

test3=dict(
    type=dataset_type,
    ann_file=data_root3 + '/instances_val.json',
    # for debugging top k imgs
    # select_first_k=200,
    img_prefix=data_root3 + '/imgs',
    pipeline=test_pipeline)
#totaltext

data_root4 = 'data/totaltext'
train4=dict(
    type=dataset_type,
    ann_file=data_root4 + '/instances_training.json',
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
train5=dict(
    type=dataset_type,
    ann_file=data_root5 + '/instances_training.json',
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


data = dict(
    samples_per_gpu=5,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=[train1,train3,train4,train5],
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=[test1,test3,test5],
        pipeline=test_pipeline),
    test=dict(
         type='UniformConcatDataset',
        datasets=[test1,test3,test5],
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='hmean-iou')
#train1,train3,train4,
#test1,test3,



# dataset_type = 'TextDetDataset'
# img_prefix1 = 'data/mydata/imgs'
# train_anno_file1 = 'data/mydata/instances_training.txt'
# train1 = dict(
#     type=dataset_type,
#     img_prefix=img_prefix1,
#     ann_file=train_anno_file1,
#     loader=dict(
#         type='HardDiskLoader',
#         repeat=4,
#         parser=dict(
#             type='LineJsonParser',
#             keys=['file_name', 'height', 'width', 'annotations'])),
#     pipeline=train_pipeline,
#     test_mode=False)
# #testocr
# img_prefix2 = 'data/textocr/imgs'
# train_anno_file2 = 'data/textocr/instances_training.txt'
# train2 = dict(
#     type=dataset_type,
#     img_prefix=img_prefix2,
#     ann_file=train_anno_file2,
#     loader=dict(
#         type='HardDiskLoader',
#         repeat=4,
#         parser=dict(
#             type='LineJsonParser',
#             keys=['file_name', 'height', 'width', 'annotations'])),
#     pipeline=train_pipeline,
#     test_mode=False)
# #testocr

# #totaltext
# img_prefix3 = 'data/totaltext/imgs'
# train_anno_file3 = 'data/totaltext/instances_training.txt'
# train3 = dict(
#     type=dataset_type,
#     img_prefix=img_prefix3,
#     ann_file=train_anno_file3,
#     loader=dict(
#         type='HardDiskLoader',
#         repeat=4,
#         parser=dict(
#             type='LineJsonParser',
#             keys=['file_name', 'height', 'width', 'annotations'])),
#     pipeline=train_pipeline,
#     test_mode=False)


# img_prefix4 = 'data/icdar2015/imgs'
# train_anno_file4 = 'data/icdar2015/instances_training.txt'
# train4 = dict(
#     type=dataset_type,
#     img_prefix=img_prefix4,
#     ann_file=train_anno_file4,
#     loader=dict(
#         type='HardDiskLoader',
#         repeat=4,
#         parser=dict(
#             type='LineJsonParser',
#             keys=['file_name', 'height', 'width', 'annotations'])),
#     pipeline=train_pipeline,
#     test_mode=False)

# img_prefix1 = 'data/mydata/imgs'
# test_anno_file1 = 'data/mydata/instances_test.txt'
# test1 = dict(
#     type=dataset_type,
#     img_prefix=img_prefix1,
#     ann_file=test_anno_file1,
#     loader=dict(
#         type='HardDiskLoader',
#         repeat=1,
#         parser=dict(
#             type='LineJsonParser',
#             keys=['file_name', 'height', 'width', 'annotations'])),
#     pipeline=test_pipeline,
#     test_mode=True)

# img_prefix2 = 'data/textocr/imgs'
# test_anno_file2 = 'data/textocr/instances_val.txt'
# train5= dict(
#     type=dataset_type,
#     img_prefix=img_prefix2,
#     ann_file=test_anno_file2,
#     loader=dict(
#         type='HardDiskLoader',
#         repeat=1,
#         parser=dict(
#             type='LineJsonParser',
#             keys=['file_name', 'height', 'width', 'annotations'])),
#     pipeline=train_pipeline,
#     test_mode=True)


# img_prefix3 = 'data/totaltext/imgs'
# test_anno_file3 = 'data/totaltext/instances_test.txt'
# test3= dict(
#     type=dataset_type,
#     img_prefix=img_prefix3,
#     ann_file=test_anno_file3,
#     loader=dict(
#         type='HardDiskLoader',
#         repeat=1,
#         parser=dict(
#             type='LineJsonParser',
#             keys=['file_name', 'height', 'width', 'annotations'])),
#     pipeline=test_pipeline,
#     test_mode=True)




# dataset_type = 'IcdarDataset'
# data_root = 'data/mydata/'
# data = dict(
#     samples_per_gpu=5,
#     workers_per_gpu=4,
#     val_dataloader=dict(samples_per_gpu=1),
#     test_dataloader=dict(samples_per_gpu=1),
#     train=dict(
#         type='UniformConcatDataset',
#         datasets=[train1, train2, train3, train4, train5],
#         pipeline=train_pipeline),
#     val=dict(
#          type='UniformConcatDataset',
#         datasets=[test1,test3,test4],
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#        datasets=[test1,test3,test4],
#         pipeline=test_pipeline))
# evaluation = dict(interval=1, metric='hmean-iou')