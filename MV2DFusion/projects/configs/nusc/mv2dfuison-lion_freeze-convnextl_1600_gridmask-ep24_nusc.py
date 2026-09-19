_base_ = ['./mv2dfusion-fsd_freeze-convnextl_1600_gridmask-ep24_nusc.py']

# 删除 FSDv2 插件导入，避免继续加载 FSDv2 依赖
plugin_dir = [
    'projects/mmdet3d_plugin/',
]

lion_pts_ckpt = '/media/shs/0b404475-9f2a-462b-9440-2e145666c221/wy_code_space/LION/weights/checkpoint_epoch_36_nus_mamba_spconv2.pth'
lion_cfg_file = '/media/shs/0b404475-9f2a-462b-9440-2e145666c221/wy_code_space/LION/tools/cfgs/lion_models/lion_mamba_nusc_8x_1f_1x_one_stride_128dim.yaml'

model = dict(

    loss_weight_pts=0.0,  # LION 冻结时不将其无梯度 loss 计入总 loss

    pts_backbone=dict(
        _delete_=True,
        type='LION3DDetectorAdapter',
        lion_root='../LION',
        cfg_file=lion_cfg_file,
        ckpt_path=lion_pts_ckpt,
        strict_ckpt=True,
        freeze=True,
        norm_eval=True,
        output_channels=128,

        restore_intensity=True,
    ),
    pts_query_generator=dict(
        in_channels=128,
        hidden_channel=128,
        pts_use_cat=False,
    ),
    fusion_bbox_head=dict(
        # x, y, z, log(w), log(l), log(h), sin(yaw), cos(yaw), vx, vy
        code_weights=[
            2.0, 2.0, 1.0,       # 定位
            1.5, 1.5, 1.5,       # 尺寸
            2.0, 2.0,            # 朝向
            2.0, 2.0,            # 速度
        ],
        loss_bbox=dict(
            type='L1Loss',
            loss_weight=0.5,
        ),
        # transformer=dict(
        #     decoder=dict(
        #         num_layers=6,
        #     ),
        # ),
    ),
)

optimizer = dict(
    lr=5e-5,
    # paramwise_cfg=dict(
    #     custom_keys={
    #         # 前6个预训练解码层：1e-5
    #         'fusion_bbox_head.transformer.decoder.layers': dict(
    #             lr_mult=0.1,
    #         ),

    #         # 第7个随机初始化解码层：1e-4
    #         # 更具体、更长的参数名会优先匹配
    #         'fusion_bbox_head.transformer.decoder.layers.6': dict(
    #             lr_mult=1.0,
    #         ),
    #     },
    # ),
)
