# Copyright (c) Phigent Robotics. All rights reserved.
from ...ops import TRTBEVPoolv2
from .bevdet import BEVDet
from .bevdepth import BEVDepth
from .bevdepth4d import BEVDepth4D
from .bevstereo4d import BEVStereo4D
from mmdet3d.models import DETECTORS
from mmdet3d.models.builder import build_head, build_backbone,build_neck
import torch.nn.functional as F
from mmdet3d.core import bbox3d2result
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from ...ops import nearest_assign
from ..vmamba.vmamba import *
from .custombevdepth4d import CustomOCCDepth4D, CustomOCCDepth4D_FO, CustomOCCDepth4D_Stereo_FO
from ..backbones.fpn3d import FPN3D

# pool = ThreadPool(processes=4)  # �����̳߳�

# for pano
grid_config_occ = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 1.0],
}        
# det
det_class_name = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
    'barrier']

# occ
occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

det_ind = [2, 3, 4, 5, 6, 7, 9, 10]
occ_ind = [5, 3, 0, 4, 6, 7, 2, 1]
detind2occind = {
    0:4,
    1:10,
    2:9,
    3:3,
    4:5,
    5:2,
    6:6,
    7:7,
    8:8,
    9:1,
}



inst_occ = np.ones([200, 200, 16])*0

import torch
X1, Y1, Z1 = 200, 200, 16
coords_x = torch.arange(X1).float()
coords_y = torch.arange(Y1).float()
coords_z = torch.arange(Z1).float()
coords = torch.stack(torch.meshgrid([coords_x, coords_y, coords_z])).permute(1, 2, 3, 0)  # W, H, D, 3
# coords = coords.cpu().numpy()
st = [grid_config_occ['x'][0], grid_config_occ['y'][0], grid_config_occ['z'][0]]
sx = [grid_config_occ['x'][2], grid_config_occ['y'][2], 0.4]

@DETECTORS.register_module()
class OCCMambaLite(BEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 depth_loss_type = 'loss_occlusion',
                 **kwargs):
        super(OCCMambaLite, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.depth_loss_type = depth_loss_type

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        gt_depth = kwargs['gt_depth']       

        occ_bev_feature = img_feats[0]
        if self.depth_loss_type == 'loss_occlusion':
            loss_depth = self.img_view_transformer.get_depth_loss_occlusion(gt_depth, depth)
        elif self.depth_loss_type == 'loss_raw':
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        else:
            raise NotImplementedError(
                f"Depth loss type is not implemented. "
            )
        losses['loss_depth'] = loss_depth

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)

        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, depth = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)

        
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        if not hasattr(self.occ_head, "get_occ_gpu"):
            occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        else:
            occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs
    

@DETECTORS.register_module()
class OCCMambaLite_FOBA(BEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 depth_loss_type = 'loss_occlusion',
                 **kwargs):
        super(OCCMambaLite_FOBA, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.depth_loss_type = depth_loss_type
    
    def bev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x0 = x[0]
            x1 = x[1]
            x2 = x[2]
        return x0, x1, x2

    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        """ Extract features of images.
        img_inputs:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N, D, fH, fW)
        """
        img_inputs = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(img_inputs[0])    # x: (B, N, C, fH, fW)
        x, depth = self.img_view_transformer([x] + img_inputs[1:7])
        # x: (B, C, Dy, Dx), BEV feature
        # depth: (B*N, D, fH, fW)
        x, x_background, x_foreground = self.bev_encoder(x)
        return [x, x_background, x_foreground], depth
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        gt_depth = kwargs['gt_depth']       

        occ_bev_feature = img_feats[0]
        if self.depth_loss_type == 'loss_occlusion':
            loss_depth = self.img_view_transformer.get_depth_loss_occlusion(gt_depth, depth)
        elif self.depth_loss_type == 'loss_raw':
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        else:
            raise NotImplementedError(
                f"Depth loss type is not implemented. "
            )
        losses['loss_depth'] = loss_depth

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        loss_occ = self.forward_occ_train(occ_bev_feature, img_feats[1:], voxel_semantics, mask_camera)
        losses.update(loss_occ)

        return losses

    def forward_occ_train(self, img_feats, foba_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            foba_feats: list[(B, C, Dz, Dy, Dx/2), (B, C, Dz, Dy, Dx/2)]
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs_tri, outs_ba, outs_fo = self.occ_head(img_feats, foba_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs_tri,  # (B, Dx, Dy, Dz, 3)
            outs_ba,  # (B, Dx, Dy, Dz, 9)
            outs_fo,  # (B, Dx, Dy, Dz, 9)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, depth = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)
        
        
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature,img_feats[1:],img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, foba_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs_tri, outs_ba, outs_fo = self.occ_head(img_feats,foba_feats)
        if not hasattr(self.occ_head, "get_occ_gpu"):
            occ_preds = self.occ_head.get_occ(outs_tri, outs_ba, outs_fo, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        else:
            occ_preds = self.occ_head.get_occ_gpu(outs_tri, outs_ba, outs_fo, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs

@DETECTORS.register_module()
class OCCMambaLite_FOBA_V2(BEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 depth_loss_type = 'loss_occlusion',
                 **kwargs):
        super(OCCMambaLite_FOBA_V2, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.depth_loss_type = depth_loss_type

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        gt_depth = kwargs['gt_depth']       

        occ_bev_feature = img_feats[0]
        if self.depth_loss_type == 'loss_occlusion':
            loss_depth = self.img_view_transformer.get_depth_loss_occlusion(gt_depth, depth)
        elif self.depth_loss_type == 'loss_raw':
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        else:
            raise NotImplementedError(
                f"Depth loss type is not implemented. "
            )
        losses['loss_depth'] = loss_depth

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)

        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs, outs_tri = self.occ_head(img_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs, # (B, Dx, Dy, Dz, n_cls)
            outs_tri,  # (B, Dx, Dy, Dz, 3)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, depth = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)
        
        
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs, _ = self.occ_head(img_feats)
        if not hasattr(self.occ_head, "get_occ_gpu"):
            occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        else:
            occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs


@DETECTORS.register_module()
class OCCMambaLite_Mask(BEVDet):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 depth_loss_type = 'loss_occlusion',
                 **kwargs):
        super(OCCMambaLite_Mask, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.depth_loss_type = depth_loss_type
        self.matching_pairs = None
        
    
    def bev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x0 = x[0]
            x1 = x[1]
        return x0, x1
        
    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        """ Extract features of images.
        img_inputs:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N, D, fH, fW)
        """
        img_inputs = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(img_inputs[0])    # x: (B, N, C, fH, fW)
        x, depth = self.img_view_transformer([x] + img_inputs[1:7])
        # x: (B, C, Dy, Dx), BEV feature
        # depth: (B*N, D, fH, fW)
        x, bev_mask_feats = self.bev_encoder(x)
        return [x, bev_mask_feats], depth

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        gt_depth = kwargs['gt_depth']       

        occ_bev_feature = img_feats[0]
        if self.depth_loss_type == 'loss_occlusion':
            loss_depth = self.img_view_transformer.get_depth_loss_occlusion(gt_depth, depth)
        elif self.depth_loss_type == 'loss_raw':
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        else:
            raise NotImplementedError(
                f"Depth loss type is not implemented. "
            )
        losses['loss_depth'] = loss_depth

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        loss_occ = self.forward_occ_train(occ_bev_feature, img_feats[1], voxel_semantics, mask_camera)
        losses.update(loss_occ)

        return losses

    def forward_occ_train(self, img_feats, bev_mask_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        masks_lst_Nq, mask_cls_lst_Nq = self.occ_head(img_feats, bev_mask_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ, matching_pairs = self.occ_head.loss2( masks_lst_Nq, 
                                    mask_cls_lst_Nq,
                                    voxel_semantics,
                                    mask_camera # (B, Dx, Dy, Dz)
        )
        self.matching_pairs = matching_pairs
        torch.save(self.matching_pairs, 'matching_pairs.pth')  

        return loss_occ
    

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, depth = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)
        

        
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_feats[1], img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, bev_mask_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        masks_lst_Nq, mask_cls_lst_Nq = self.occ_head(img_feats, bev_mask_feats)
 
        self.matching_pairs = torch.load('matching_pairs_test.pth')
        row_indices = self.matching_pairs[0]
        col_indices = self.matching_pairs[1]

        sorted_indices = torch.argsort(col_indices) 
        sorted_row_indices = row_indices[sorted_indices].to(masks_lst_Nq.device)    

        masks_lst_Nq = masks_lst_Nq[:, sorted_row_indices, :, :, :]  
        mask_cls_lst_Nq = mask_cls_lst_Nq[:, sorted_row_indices, :] 
        
        if not hasattr(self.occ_head, "get_occ_gpu"):
            occ_preds = self.occ_head.get_occ(masks_lst_Nq, mask_cls_lst_Nq, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        else:
            occ_preds = self.occ_head.get_occ_gpu(masks_lst_Nq, mask_cls_lst_Nq, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs


@DETECTORS.register_module()
class OCCMambaLite_FO(BEVDet):
    def __init__(self,
                 occ_head=None,
                 occ_fpn3d = None,
                 upsample=False,
                 depth_loss_type = 'loss_occlusion',
                 **kwargs):
        super(OCCMambaLite_FO, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.occ_fpn3d =  build_backbone(occ_fpn3d)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.depth_loss_type = depth_loss_type
        self.mask_net = nn.Sequential(  
            nn.Conv3d(64, 64, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm3d(64),  
            nn.ReLU(inplace=True), 
            nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1), 
            nn.Sigmoid()  
        ) 
        self.conv3d = nn.Sequential(   
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, groups=4),  # 分组卷积  
            nn.BatchNorm3d(64),  
            nn.ReLU(inplace=True),  
        )  
        
    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        """Extract features from images and points."""
        """
        points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
        img_inputs:
                imgs:  (B, N_views, 3, H, W)        
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        """
        img_feats, occ_feats, depth, mask_fo = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = None
        return img_feats, occ_feats, pts_feats, depth, mask_fo
        
    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        """ Extract features of images.
        img_inputs:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
        Returns:
            x: [(B, C', H', W'), ]
            x_occ: (B, C, Dz, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        img_inputs = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(img_inputs[0])    # x: (B, N, C, fH, fW)
        x, x_occ, depth = self.img_view_transformer([x] + img_inputs[1:7])
        # x: (B, C, Dy, Dx), BEV feature
        # depth: (B*N, D, fH, fW)
        # x_occ: (B, C, Dz, Dy, Dx)
        x = self.bev_encoder(x)
        x_occ, mask_fo = self.occ_fo_choose(x_occ)
        return [x], x_occ, depth, mask_fo

    def occ_fo_choose(self,
                      occ_feature=None,
                      ):
        occ_feature_enhanced = self.conv3d(occ_feature)
        mask = self.mask_net(occ_feature_enhanced)
        occ_feature_enhanced = occ_feature_enhanced * mask
        x_occ = self.occ_fpn3d(occ_feature_enhanced)
        
        return x_occ, mask
    

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, occ_feats, pts_feats, depth, mask_fo = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        gt_depth = kwargs['gt_depth']       

        occ_bev_feature = img_feats[0]
        if self.depth_loss_type == 'loss_occlusion':
            loss_depth = self.img_view_transformer.get_depth_loss_occlusion(gt_depth, depth)
        elif self.depth_loss_type == 'loss_raw':
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        else:
            raise NotImplementedError(
                f"Depth loss type is not implemented. "
            )
        losses['loss_depth'] = loss_depth

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        loss_occ = self.forward_occ_train(occ_bev_feature, occ_feats, voxel_semantics, mask_camera, mask_fo)
        losses.update(loss_occ)

        return losses

    def forward_occ_train(self, img_feats, occ_feats, voxel_semantics, mask_camera, mask_fo):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            occ_feats: (B, C, Dz, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
            mask_fo: (B, 1, Dz, Dy, Dx)
        Returns:
        """
        mask_fo = mask_fo.squeeze(1).permute(0,3,2,1)    # (B, Dx, Dy, Dz)
        outs, occ_fo_outs, occ_final_outs = self.occ_head(img_feats, occ_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            occ_fo_outs,
            occ_final_outs,
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
            mask_fo,        # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, occ_feats, pts_feats, depth, mask_fo = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)
        

        
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, occ_feats, img_metas, mask_fo)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, occ_feats, img_metas=None, mask_fo=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        mask_fo = mask_fo.squeeze(1).permute(0,3,2,1)    # (B, Dx, Dy, Dz)
        outs = self.occ_head(img_feats, occ_feats, training_process = False)
        if not hasattr(self.occ_head, "get_occ_gpu"):
            occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        else:
            occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, occ_feats, pts_feats, depth, mask_fo = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
    
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature, occ_feats)
        return outs




@DETECTORS.register_module()
class OCCMambaLite4D_FO(CustomOCCDepth4D_FO):
    def __init__(self,
                 occ_head=None,
                 occ_fpn3d = None,
                 upsample=False,
                 depth_loss_type = 'loss_occlusion',
                 **kwargs):
        super(OCCMambaLite4D_FO, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.occ_fpn3d =  build_backbone(occ_fpn3d)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.depth_loss_type = depth_loss_type
        
    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        """Extract features from images and points."""
        """
        points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
        img_inputs:
                imgs:  (B, N_views, 3, H, W)        
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        """
        img_feats, occ_feats, depth, mask_fo = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = None
        return img_feats, occ_feats, pts_feats, depth, mask_fo
        

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, occ_feats, pts_feats, depth, mask_fo = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        gt_depth = kwargs['gt_depth']       

        occ_bev_feature = img_feats[0]
        if self.depth_loss_type == 'loss_occlusion':
            loss_depth = self.img_view_transformer.get_depth_loss_occlusion(gt_depth, depth)
        elif self.depth_loss_type == 'loss_raw':
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        else:
            raise NotImplementedError(
                f"Depth loss type is not implemented. "
            )
        losses['loss_depth'] = loss_depth

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        loss_occ = self.forward_occ_train(occ_bev_feature, occ_feats, voxel_semantics, mask_camera, mask_fo)
        losses.update(loss_occ)

        return losses

    def forward_occ_train(self, img_feats, occ_feats, voxel_semantics, mask_camera, mask_fo):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            occ_feats: (B, C, Dz, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
            mask_fo: (B, 1, Dz, Dy, Dx)
        Returns:
        """
        mask_fo = mask_fo.squeeze(1).permute(0,3,2,1)    # (B, Dx, Dy, Dz)
        outs, occ_fo_outs, occ_final_outs = self.occ_head(img_feats, occ_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            occ_fo_outs,
            occ_final_outs,
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
            mask_fo,        # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, occ_feats, pts_feats, depth, mask_fo = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)
        

        
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, occ_feats, img_metas, mask_fo)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, occ_feats, img_metas=None, mask_fo=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        mask_fo = mask_fo.squeeze(1).permute(0,3,2,1)    # (B, Dx, Dy, Dz)
        outs = self.occ_head(img_feats, occ_feats, training_process = False)
        if not hasattr(self.occ_head, "get_occ_gpu"):
            occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        else:
            occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs



@DETECTORS.register_module()
class OCCMambaLite4D_Stereo_FO(CustomOCCDepth4D_Stereo_FO):
    def __init__(self,
                 occ_head=None,
                 occ_fpn3d = None,
                 upsample=False,
                 depth_loss_type = 'loss_occlusion',
                 **kwargs):
        super(OCCMambaLite4D_Stereo_FO, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.occ_fpn3d =  build_backbone(occ_fpn3d)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.depth_loss_type = depth_loss_type
        
    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        """Extract features from images and points."""
        """
        points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
        img_inputs:
                imgs:  (B, N_views, 3, H, W)        
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        """
        img_feats, occ_feats, depth, mask_fo = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        pts_feats = None
        return img_feats, occ_feats, pts_feats, depth, mask_fo
        

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, occ_feats, pts_feats, depth, mask_fo = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        gt_depth = kwargs['gt_depth']       

        occ_bev_feature = img_feats[0]
        if self.depth_loss_type == 'loss_occlusion':
            loss_depth = self.img_view_transformer.get_depth_loss_occlusion(gt_depth, depth)
        elif self.depth_loss_type == 'loss_raw':
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        else:
            raise NotImplementedError(
                f"Depth loss type is not implemented. "
            )
        losses['loss_depth'] = loss_depth

        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        loss_occ = self.forward_occ_train(occ_bev_feature, occ_feats, voxel_semantics, mask_camera, mask_fo)
        losses.update(loss_occ)

        return losses

    def forward_occ_train(self, img_feats, occ_feats, voxel_semantics, mask_camera, mask_fo):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            occ_feats: (B, C, Dz, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
            mask_fo: (B, 1, Dz, Dy, Dx)
        Returns:
        """
        mask_fo = mask_fo.squeeze(1).permute(0,3,2,1)    # (B, Dx, Dy, Dz)
        outs, occ_fo_outs, occ_final_outs = self.occ_head(img_feats, occ_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            occ_fo_outs,
            occ_final_outs,
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
            mask_fo,        # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, occ_feats, pts_feats, depth, mask_fo = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)
 
        
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, occ_feats, img_metas, mask_fo)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, occ_feats, img_metas=None, mask_fo=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        mask_fo = mask_fo.squeeze(1).permute(0,3,2,1)    # (B, Dx, Dy, Dz)
        outs = self.occ_head(img_feats, occ_feats, training_process = False)
        if not hasattr(self.occ_head, "get_occ_gpu"):
            occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        else:
            occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs







@DETECTORS.register_module()
class OCCMamba4D(BEVStereo4D):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 depth_loss_type='loss_occlusion',
                 **kwargs):
        super(OCCMamba4D, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.depth_loss_type = depth_loss_type

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
        losses = dict()
        if self.depth_loss_type == 'loss_occlusion':
            loss_depth = self.img_view_transformer.get_depth_loss_occlusion(gt_depth, depth)
        elif self.depth_loss_type == 'loss_raw':
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        else:
            raise NotImplementedError(
                f"Depth loss type is not implemented. "
            )
        losses['loss_depth'] = loss_depth
        
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        loss_occ = self.forward_occ_train(img_feats[0], voxel_semantics, mask_camera)
        losses.update(loss_occ)
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, _ = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)

        occ_list = self.simple_test_occ(img_feats[0], img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        # occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs
    

@DETECTORS.register_module()
class OCCMamba4D_Vanilla(CustomOCCDepth4D):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 depth_loss_type='loss_occlusion',
                 **kwargs):
        super(OCCMamba4D_Vanilla, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample
        self.depth_loss_type = depth_loss_type

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
        losses = dict()
        if self.depth_loss_type == 'loss_occlusion':
            loss_depth = self.img_view_transformer.get_depth_loss_occlusion(gt_depth, depth)
        elif self.depth_loss_type == 'loss_raw':
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        else:
            raise NotImplementedError(
                f"Depth loss type is not implemented. "
            )
        losses['loss_depth'] = loss_depth
        
        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)
        loss_occ = self.forward_occ_train(img_feats[0], voxel_semantics, mask_camera)
        losses.update(loss_occ)
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, _ = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)

        occ_list = self.simple_test_occ(img_feats[0], img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        # occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs



@DETECTORS.register_module()
class BEVDepthOCC(BEVDepth):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 **kwargs):
        super(BEVDepthOCC, self).__init__(**kwargs)
        self.occ_head = build_head(occ_head)
        self.pts_bbox_head = None
        self.upsample = upsample

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        losses = dict()
        gt_depth = kwargs['gt_depth']   # (B, N_views, img_H, img_W)
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses['loss_depth'] = loss_depth

        voxel_semantics = kwargs['voxel_semantics']     # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']     # (B, Dx, Dy, Dz)

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        loss_occ = self.forward_occ_train(occ_bev_feature, voxel_semantics, mask_camera)
        losses.update(loss_occ)
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        outs = self.occ_head(img_feats)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss(
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, _, _ = self.extract_feat(
            points, img_inputs=img, img_metas=img_metas, **kwargs)

        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ(occ_bev_feature, img_metas)    # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        outs = self.occ_head(img_feats)
        # occ_preds = self.occ_head.get_occ(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        occ_preds = self.occ_head.get_occ_gpu(outs, img_metas)      # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats, pts_feats, depth = self.extract_feat(
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)
        occ_bev_feature = img_feats[0]
        if self.upsample:
            occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
                                            mode='bilinear', align_corners=True)
        outs = self.occ_head(occ_bev_feature)
        return outs

