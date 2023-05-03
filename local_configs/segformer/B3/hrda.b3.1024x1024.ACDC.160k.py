_base_ = [
    '../_base_/models/daformer_sepaspp_mitb3.py',
    '../../_base_/datasets/ACDC_512x512_dwnsample_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
] #todo: modify to HRDA

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='HRDAEncoderDecoder',
    pretrained='pretrained/mit_b3.pth',
    backbone=dict(
        type='mit_b3',
        style='pytorch'),
    decode_head=dict(
        type='HRDAHead',
        # Use the DAFormer decoder for each scale.
        single_scale_head='DAFormerHead',
        # Learn a scale attention for each class channel of the prediction.
        attention_classwise=True,
        # Set the detail loss weight $\lambda_d=0.1$.
        hr_loss_weight=0.1),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'))
    test_cfg=dict(mode='slide', crop_size=(1024//2,1024//2), stride=(768//2,768//2))) # modified

# data
data = dict(samples_per_gpu=1)
evaluation = dict(interval=4000, metric='mIoU')

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
