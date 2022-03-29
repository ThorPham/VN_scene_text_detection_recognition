_base_ = [
    '../../_base_/default_runtime.py', '../../_base_/recog_models/satrn.py'
]
load_from = 'satrnet-full/epoch_5.pth'
label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', with_unknown=True)

model = dict(
    type='SATRN',
    backbone=dict(type='ShallowCNN', input_channels=3, hidden_dim=512),
    encoder=dict(
        type='SatrnEncoder',
        n_layers=12,
        n_head=8,
        d_k=512 // 8,
        d_v=512 // 8,
        d_model=512,
        n_position=100,
        d_inner=512 * 4,
        dropout=0.1),
    decoder=dict(
        type='TFDecoder',
        n_layers=6,
        d_embedding=512,
        n_head=8,
        d_model=512,
        d_inner=512 * 4,
        d_k=512 // 8,
        d_v=512 // 8),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=25)

# optimizer
optimizer = dict(type='Adam', lr=3e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 8

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MyAuggemt',p=0.5),
    dict(
        type='ResizeOCR',
        height=48,
        min_width=160,
        max_width=160,
        keep_aspect_ratio=True,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio',
            'resize_shape'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=48,
                min_width=160,
                max_width=160,
                keep_aspect_ratio=False,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(type='NormalizeOCR', **img_norm_cfg),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio',
                    'resize_shape'
                ]),
        ])
]




# train_img_prefix1 = train_prefix + \
#     'SynthText/synthtext/SynthText_patch_horizontal'
# train_img_prefix2 = train_prefix + 'Syn90k/mnt/ramdisk/max/90kDICT32px'

# train_ann_file1 = train_prefix + 'SynthText/label.lmdb'
# train_ann_file2 = train_prefix + 'Syn90k/label.lmdb'

# train1 = dict(
#     type=dataset_type,
#     img_prefix=train_img_prefix1,
#     ann_file=train_ann_file1,
#     loader=dict(
#         type='LmdbLoader',
#         repeat=1,
#         parser=dict(
#             type='LineStrParser',
#             keys=['filename', 'text'],
#             keys_idx=[0, 1],
#             separator=' ')),
#     pipeline=None,
#     test_mode=False)

# train2 = {key: value for key, value in train1.items()}
# train2['img_prefix'] = train_img_prefix2
# train2['ann_file'] = train_ann_file2
dataset_type = 'OCRDataset'


train_prefix = 'data_ocr/mixture/'
train_img_prefix1 = train_prefix + 'my_ocr_data/'
train_img_prefix2 = train_prefix + 'ICDAR_2013/'
train_img_prefix3 = train_prefix + 'ICDAR_2015/'
train_img_prefix4 = train_prefix + 'coco_text/'
train_img_prefix5 = train_prefix + 'textocr/'
train_img_prefix6 = train_prefix + 'III5K/'
train_img_prefix7 = train_prefix + 'my_ocr_data/'
train_img_prefix8 = train_prefix + 'sysdata/'
train_img_prefix9 = train_prefix + 'new/'
train_img_prefix10 = train_prefix + 'SynthText_Add/'
train_img_prefix11 = train_prefix + 'gen-new/'
train_img_prefix12 = train_prefix + 'textocr/'

train_ann_file1 = train_prefix + 'my_ocr_data/train.txt'
train_ann_file2 = train_prefix + 'ICDAR_2013/train_label.txt'
train_ann_file3 = train_prefix + 'ICDAR_2015/train_label.txt'
train_ann_file4 = train_prefix + 'coco_text/train_label.txt'
train_ann_file5 = train_prefix + 'textocr/train_label.txt'
train_ann_file6 = train_prefix + 'III5K/train_label.txt'
train_ann_file7 = train_prefix + 'my_ocr_data/test.txt'
train_ann_file8 = train_prefix + 'sysdata/gt.txt'
train_ann_file9 = train_prefix + 'new/gt.txt'
train_ann_file10 = train_prefix + 'SynthText_Add/gt.txt'
train_ann_file11 = train_prefix + 'gen-new/gt.txt'
train_ann_file12 = train_prefix + 'textocr/val_label.txt'


train1 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix1,
    ann_file=train_ann_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train2 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix2,
    ann_file=train_ann_file2,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train3 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix3,
    ann_file=train_ann_file3,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train4 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix4,
    ann_file=train_ann_file4,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)


train5 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix5,
    ann_file=train_ann_file5,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train6 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix6,
    ann_file=train_ann_file6,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)


train7 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix7,
    ann_file=train_ann_file7,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train8 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix8,
    ann_file=train_ann_file8,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)



train9 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix9,
    ann_file=train_ann_file9,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train10 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix10,
    ann_file=train_ann_file10,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train11 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix11,
    ann_file=train_ann_file11,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train12 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix12,
    ann_file=train_ann_file12,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)
# train2 = {key: value for key, value in train1.items()}
# train2['img_prefix'] = train_img_prefix2
# train2['ann_file'] = train_ann_file2

# train3 = {key: value for key, value in train1.items()}
# train3['img_prefix'] = train_img_prefix3
# train3['ann_file'] = train_img_prefix3

test_prefix = 'data_ocr/mixture/'
test_img_prefix1 = test_prefix + 'my_ocr_data/'
test_img_prefix2 = test_prefix + 'ICDAR_2013/'
test_img_prefix3 = test_prefix + 'ICDAR_2015/'
test_img_prefix4 = test_prefix + 'III5K/'
test_img_prefix5 = test_prefix + 'svt/'
test_img_prefix6 = test_prefix + 'textocr/'

test_ann_file1 = test_prefix + 'my_ocr_data/test.txt'
test_ann_file2 = test_prefix + 'ICDAR_2013/test_label_1015.txt'
test_ann_file3 = test_prefix + 'ICDAR_2015/test_label.txt'
test_ann_file4 = test_prefix + 'III5K/test_label.txt'
test_ann_file5 = test_prefix + 'svt/test_label.txt'
test_ann_file6 = test_prefix + 'textocr/val_label.txt'

test1 = dict(
    type=dataset_type,
    img_prefix=test_img_prefix1,
    ann_file=test_ann_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

test2 = dict(
    type=dataset_type,
    img_prefix=test_img_prefix2,
    ann_file=test_ann_file2,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

test3 = dict(
    type=dataset_type,
    img_prefix=test_img_prefix3,
    ann_file=test_ann_file3,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

test4 = dict(
    type=dataset_type,
    img_prefix=test_img_prefix4,
    ann_file=test_ann_file4,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

test5 = dict(
    type=dataset_type,
    img_prefix=test_img_prefix5,
    ann_file=test_ann_file5,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

test6 = dict(
    type=dataset_type,
    img_prefix=test_img_prefix6,
    ann_file=test_ann_file6,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)
# test2 = {key: value for key, value in test1.items()}
# test2['img_prefix'] = test_img_prefix2
# test2['ann_file'] = test_ann_file2

# test3 = {key: value for key, value in test1.items()}
# test3['img_prefix'] = test_img_prefix3
# test3['ann_file'] = test_ann_file3


data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4),
    train=dict(
        type='UniformConcatDataset',
        datasets=[train1, train2,train3,train4,train5,train6,train7,train8,train9,train10,train11,train12],
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=[test1, test2, test3,test4, test5, test6],
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=[test1, test2, test3,test4, test5, test6],
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')
