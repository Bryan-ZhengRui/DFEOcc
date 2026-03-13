_base_ = ['../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
          '../../../mmdetection3d/configs/_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    # 'z': [-1, 5.4, 6.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 64

# workflow = [('train', 8), ('val', 1)]
multi_adj_frame_id_cfg = (1, 1+1, 1)


model = dict(
    type='OCCMambaLite4D_Stereo_FO',
    align_after_view_transfromation=False,
    depth_loss_type='loss_raw',
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        with_cp=True,
        style='pytorch',
        pretrained='torchvision://resnet50',
        ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer_FO_Stereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        sid=False,
        collapse_z=True,
        downsample=16,
        loss_depth_weight=0.1),
    occ_fpn3d = dict(
        type='FPN3D_Mamba_4D',
        in_channels = 64 * (len(range(*multi_adj_frame_id_cfg))+1), 
        hidden_channels = 80,
        out_channels = 64, 
        ),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    pre_process=dict(
        type='CustomResNet', 
        numC_input=numC_Trans,
        num_layer=[1, ],
        num_channels=[numC_Trans, ],
        stride=[1, ],
        backbone_output_ids=[0, ]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS_Mamba_FO',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    occ_head=dict(
        type='BEVOCCHead_FO',
        in_dim=256,
        out_dim=512,    
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_balance=True,
        loss_mask_fo=dict(
            type='BinaryFocalLoss',
            alpha=0.75, 
            gamma=2.0,
        ),
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0
        ),
        loss_occ_fo=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0
        ),
        loss_occ_final=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0
        ),
        loss_occ_weight=0.6,
        loss_occ_fo_weight=10.0,
        loss_occ_final_weight=3.0,
        loss_mask_fo_weight=16.0,
    )
)

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = './data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root +'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file= data_root +'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=5e-5, weight_decay=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=0.001,
    step=[6,12,18,24,30],
    gamma=0.6,
    )
runner = dict(type='EpochBasedRunner', max_epochs=35)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]


# fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=1, start=20, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=5)