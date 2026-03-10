# Copyright (c) OpenMMLab. All rights reserved.
import torch, math
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
import numpy as np
from mmdet3d.models.builder import HEADS, build_loss
from ..losses.semkitti_loss import sem_scal_loss, geo_scal_loss
from ..losses.lovasz_softmax import lovasz_softmax
from scipy.optimize import linear_sum_assignment  



nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
])

foreground_class = [2,3,4,5,6,7,8,9,10]
background_class = [0,1,11,12,13,14,15,16]
free_class = [17]
# foreground_withfree_class = [2,3,4,5,6,7,8,9,10,17]


@HEADS.register_module()
class BEVOCCHead3D(BaseModule):
    def __init__(self,
                 in_dim=32,
                 out_dim=32,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ=None
                 ):
        super(BEVOCCHead3D, self).__init__()
        self.out_dim = 32
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
            in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim*2),
                nn.Softplus(),
                nn.Linear(self.out_dim*2, num_classes),
            )

        self.num_classes = num_classes
        self.use_mask = use_mask
        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
            loss_occ['class_weight'] = class_weights

        self.loss_occ = build_loss(loss_occ)

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx)

        Returns:

        """
        # (B, C, Dz, Dy, Dx) --> (B, C, Dz, Dy, Dx) --> (B, Dx, Dy, Dz, C)
        occ_pred = self.final_conv(img_feats).permute(0, 4, 3, 2, 1)
        if self.use_predicter:
            # (B, Dx, Dy, Dz, C) --> (B, Dx, Dy, Dz, 2*C) --> (B, Dx, Dy, Dz, n_cls)
            occ_pred = self.predicter(occ_pred)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            voxel_semantics = voxel_semantics.reshape(-1)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
            preds = occ_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            mask_camera = mask_camera.reshape(-1)

            if self.class_balance:
                valid_voxels = voxel_semantics[mask_camera.bool()]
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = mask_camera.sum()

            loss_occ = self.loss_occ(
                preds,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                mask_camera,        # (B*Dx*Dy*Dz, )
                avg_factor=num_total_samples
            )
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)

            if self.class_balance:
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = len(voxel_semantics)

            loss_occ = self.loss_occ(
                preds,
                voxel_semantics,
                avg_factor=num_total_samples
            )

        loss['loss_occ'] = loss_occ
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)


@HEADS.register_module()
class BEVOCCHead2D(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ=None,
                 ):
        super(BEVOCCHead2D, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes

        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
            loss_occ['class_weight'] = class_weights        # ce loss
        self.loss_occ = build_loss(loss_occ)

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx)

        Returns:

        """
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            voxel_semantics = voxel_semantics.reshape(-1)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
            preds = occ_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            mask_camera = mask_camera.reshape(-1)

            if self.class_balance:
                valid_voxels = voxel_semantics[mask_camera.bool()]
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = mask_camera.sum()

            loss_occ = self.loss_occ(
                preds,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                mask_camera,        # (B*Dx*Dy*Dz, )
                avg_factor=num_total_samples
            )
            loss['loss_occ'] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)

            if self.class_balance:
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = len(voxel_semantics)

            loss_occ = self.loss_occ(
                preds,
                voxel_semantics,
                avg_factor=num_total_samples
            )

            loss['loss_occ'] = loss_occ
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)


@HEADS.register_module()
class BEVOCCHead2D_Lova(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ=None,
                 ):
        super(BEVOCCHead2D_Lova, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes

        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
            loss_occ['class_weight'] = class_weights        # ce loss
        self.loss_occ = build_loss(loss_occ)

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx)

        Returns:

        """
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            voxel_semantics = voxel_semantics.reshape(-1)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
            preds = occ_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            mask_camera = mask_camera.reshape(-1)

            if self.class_balance:
                valid_voxels = voxel_semantics[mask_camera.bool()]
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = mask_camera.sum()

            loss_occ = self.loss_occ(
                preds,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                mask_camera,        # (B*Dx*Dy*Dz, )
                avg_factor=num_total_samples
            )
            loss['loss_occ'] = loss_occ
            loss['loss_voxel_lovasz'] = lovasz_softmax(torch.softmax(preds, dim=1), voxel_semantics)
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)

            if self.class_balance:
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = len(voxel_semantics)

            loss_occ = self.loss_occ(
                preds,
                voxel_semantics,
                avg_factor=num_total_samples
            )

            loss['loss_occ'] = loss_occ
            loss['loss_voxel_lovasz'] = lovasz_softmax(torch.softmax(preds, dim=1), voxel_semantics)
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)


@HEADS.register_module()
class BEVOCCHead2D_FOBA(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ=None,
                 weight_tri_loss=0.5,
                 weight_background_loss=0.2,
                 weight_foreground_loss=0.1
                 ):
        super(BEVOCCHead2D_FOBA, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.final_conv_ba = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.final_conv_fo = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter_tri = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, 3 * Dz),
            )
            self.predicter_background = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, 8 * Dz),
            )
            self.predicter_foreground = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, 9 * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes
        self.tri_class = 3
        self.background_class = 8
        self.foreground_class = 9
        self.weight_tri_loss = weight_tri_loss
        self.weight_background_loss = weight_background_loss
        self.weight_foreground_loss = weight_foreground_loss

        self.class_balance = class_balance
        if self.class_balance:
            # Extract corresponding class frequencies from nusc_class_frequencies for specified indices
            foreground_freqs = nusc_class_frequencies[foreground_class]  # Foreground class frequencies (9 classes)
            background_freqs = nusc_class_frequencies[background_class]  # Background class frequencies (8 classes)

            # Calculate class weights using the formula 1 / log(freq + 0.001)
            class_weights_fo = torch.from_numpy(1 / np.log(foreground_freqs + 0.001))  # Foreground weights
            class_weights_ba = torch.from_numpy(1 / np.log(background_freqs + 0.001))  # Background weights

            # Store only foreground and background weights
            self.cls_weights_fo = class_weights_fo  # Foreground weights (9 classes)
            self.cls_weights_ba = class_weights_ba  # Background weights (8 classes)

            # Calculate total frequencies for three-class (foreground, background, free)
            foreground_total_freq = nusc_class_frequencies[foreground_class].sum()  
            background_total_freq = nusc_class_frequencies[background_class].sum()  
            free_freq = nusc_class_frequencies[free_class[0]]  

            # Calculate weights for three-class case
            class_weights_tri_np = 1 / np.log(np.array([foreground_total_freq, background_total_freq, free_freq]) + 0.001)  
            class_weights_tri = torch.from_numpy(class_weights_tri_np).float()  # Three-class weights (3 classes)  
            self.cls_weights_tri = class_weights_tri  # Store three-class weights (3 classes)  

        self.loss_occ = build_loss(loss_occ)

    def forward(self, img_feats, foba_feats):
        """
        Args:
            img_feats: (B, C, Dy, Dx)   
            foba_feats: list[(B, C, Dz, Dy, Dx'), (B, C, Dz, Dy, Dx')]

        Returns:

        """
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        occ_pred_ba = self.final_conv_ba(foba_feats[0]).permute(0, 3, 2, 1)
        occ_pred_fo = self.final_conv_fo(foba_feats[1]).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*3)
            occ_pred_tri = self.predicter_tri(occ_pred)
            occ_pred_tri = occ_pred_tri.view(bs, Dx, Dy, self.Dz, self.tri_class)
            
            occ_pred_ba = self.predicter_background(occ_pred_ba)
            occ_pred_ba = occ_pred_ba.view(bs, Dx, Dy, self.Dz, self.background_class) #(B, Dx, Dy, Dz*8)

            occ_pred_fo = self.predicter_foreground(occ_pred_fo)
            occ_pred_fo = occ_pred_fo.view(bs, Dx, Dy, self.Dz, self.foreground_class) #(B, Dx, Dy, Dz*9)

        return occ_pred_tri, occ_pred_ba, occ_pred_fo

    def loss(self, occ_pred_tri, occ_pred_ba, occ_pred_fo, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred_tri: (B, Dx, Dy, Dz, 3) -> Three-class prediction for foreground/background/free
            occ_pred_ba: (B, Dx, Dy, Dz, 8) -> Fine-grained background prediction
            occ_pred_fo: (B, Dx, Dy, Dz, 9) -> Fine-grained foreground prediction
            voxel_semantics: (B, Dx, Dy, Dz) -> Raw semantic labels (18 classes)
            mask_camera: (B, Dx, Dy, Dz) -> Camera frustum region, boolean mask
        Returns:
            loss: Dictionary with three types of loss
        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()

        # Use Mask to filter out invalid voxels
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)  # (B, Dx, Dy, Dz) -> convert Mask to int
            # Initialize category mapping table (map 18-class to 3-class)
            class_map = torch.zeros(self.num_classes, dtype=torch.long, device=voxel_semantics.device)  # 18-class -> 3-class
            class_map[foreground_class] = 1  # Foreground mapped to 1
            class_map[background_class] = 0  # Background mapped to 0
            class_map[free_class] = 2        # Free space mapped to 2

            # Remap voxel_semantics to three types (background, foreground, free)
            voxel_semantics_tri = class_map[voxel_semantics]  # Three-class labels

            # ===> Change tensor shape (flatten)
            voxel_semantics_tri = voxel_semantics_tri.reshape(-1)
            mask_camera = mask_camera.reshape(-1)  # Flatten mask
            preds_tri = occ_pred_tri.reshape(-1, self.tri_class)  # Three-class prediction (B*Dx*Dy*Dz, 3)

            if self.class_balance:
                num_total_samples = 0
                for i in range(self.tri_class):
                    num_total_samples += (voxel_semantics_tri == i).sum() * self.cls_weights_tri[i]
            else:
                num_total_samples = mask_camera.sum()
            
            # ===> Compute three-class loss: loss_occ_tri
            loss_occ_tri = self.loss_occ(
                preds_tri,          # Predictions (B*Dx*Dy*Dz, 3)
                voxel_semantics_tri,  # Labels (B*Dx*Dy*Dz,)
                mask_camera,         # Mask (B*Dx*Dy*Dz,)
                avg_factor=num_total_samples
            )

            # ===> Masks for foreground (1) and background (0)
            foreground_mask = (voxel_semantics_tri == 1) & mask_camera.bool()  # Only within mask region
            background_mask = (voxel_semantics_tri == 0) & mask_camera.bool()

            # -------------------------------------------
            # Fine-grained foreground loss (9 foreground classes)
            # -------------------------------------------
            voxel_semantics = voxel_semantics.reshape(-1)
            valid_foreground_voxels = voxel_semantics[foreground_mask]  # Original labels, valid voxels for foreground only
            preds_fo = occ_pred_fo.reshape(-1, self.foreground_class)[foreground_mask]  # Foreground predictions (N, 9)

            mapping_dict = {v: i for i, v in enumerate(foreground_class)}  # Foreground class mapping
            # Use dictionary mapping to convert
            mapped_foreground_voxels = torch.tensor([mapping_dict[v.item()] for v in valid_foreground_voxels], device=preds_fo.device)  # Convert
            # Compute weighted foreground loss if class_balance is used
            if self.class_balance:
                num_total_samples_fo = 0
                for i in range(self.foreground_class):
                    num_total_samples_fo += (mapped_foreground_voxels == i).sum() * self.cls_weights_fo[i]
            else:
                num_total_samples_fo = foreground_mask.sum()

            loss_occ_fo = self.loss_occ(
                preds_fo,                     # Foreground predictions (N, 9)
                mapped_foreground_voxels,      # Foreground labels (N,)
                avg_factor=num_total_samples_fo
            )

            # -------------------------------------------
            # Fine-grained background loss (8 background classes)
            # -------------------------------------------
            valid_background_voxels = voxel_semantics[background_mask]  # Original labels, valid voxels for background only
            preds_ba = occ_pred_ba.reshape(-1, self.background_class)[background_mask]  # Background predictions (M, 8)

            mapping_dict = {v: i for i, v in enumerate(background_class)}  # Background class mapping
            # Use dictionary mapping to convert
            mapped_background_voxels = torch.tensor([mapping_dict[v.item()] for v in valid_background_voxels], device=preds_ba.device)

            # Compute weighted background loss if class_balance is used
            if self.class_balance:
                num_total_samples_ba = 0
                for i in range(self.background_class):
                    num_total_samples_ba += (mapped_background_voxels == i).sum() * self.cls_weights_ba[i]
            else:
                num_total_samples_ba = background_mask.sum()
            
            loss_occ_ba = self.loss_occ(
                preds_ba,                     # Background predictions (M, 8)
                mapped_background_voxels,      # Background labels (M,)
                avg_factor=num_total_samples_ba
            )

            loss['loss_occ_tri'] = self.weight_tri_loss*loss_occ_tri
            loss['loss_occ_fo'] = self.weight_foreground_loss*loss_occ_fo
            loss['loss_occ_ba'] = self.weight_background_loss*loss_occ_ba

            # loss['loss_voxel_lovasz'] = lovasz_softmax(torch.softmax(preds_tri, dim=1), voxel_semantics)
        else:  
            # Initialize category mapping from 18 classes to 3 classes  
            class_map = torch.zeros(self.num_classes, dtype=torch.long, device=voxel_semantics.device)  
            class_map[foreground_class] = 1  # Foreground  
            class_map[background_class] = 0  # Background  
            class_map[free_class] = 2        # Free space  

            # Remap voxel_semantics to three classes  
            voxel_semantics_tri = class_map[voxel_semantics]  # (B, Dx, Dy, Dz)  

            # Flatten tensors  
            voxel_semantics_tri = voxel_semantics_tri.reshape(-1)  
            preds_tri = occ_pred_tri.reshape(-1, self.tri_class)  

            if self.class_balance:
                num_total_samples = 0
                for i in range(self.tri_class):
                    num_total_samples += (voxel_semantics_tri == i).sum() * self.cls_weights_tri[i]
            else:
                num_total_samples = None

            # ===> Compute three-class loss: loss_occ_tri
            loss_occ_tri = self.loss_occ(
                preds_tri,          # Predictions (B*Dx*Dy*Dz, 3)
                voxel_semantics_tri,  # Labels (B*Dx*Dy*Dz,)
                avg_factor=num_total_samples
            )

            # Foreground and background masks  
            foreground_mask = (voxel_semantics_tri == 1)  
            background_mask = (voxel_semantics_tri == 0)  

            # -------------------------------------------  
            # Fine-grained foreground loss (9 foreground classes)  
            # -------------------------------------------  
            valid_foreground_voxels = voxel_semantics.reshape(-1)[foreground_mask]  
            preds_fo = occ_pred_fo.reshape(-1, self.foreground_class)[foreground_mask]  

            mapping_dict = {v: i for i, v in enumerate(foreground_class)}  # Foreground class mapping
            mapped_foreground_voxels = torch.tensor([mapping_dict[v.item()] for v in valid_foreground_voxels], device=preds_fo.device)
            if self.class_balance:
                num_total_samples_fo = 0
                for i in range(self.foreground_class):
                    num_total_samples_fo += (mapped_foreground_voxels == i).sum() * self.cls_weights_fo[i]
            else:
                num_total_samples_fo = foreground_mask.sum()

            loss_occ_fo = self.loss_occ(  
                preds_fo,  
                mapped_foreground_voxels,  
                avg_factor=num_total_samples_fo  
            )  

            # -------------------------------------------  
            # Fine-grained background loss (8 background classes)  
            # -------------------------------------------  
            valid_background_voxels = voxel_semantics.reshape(-1)[background_mask]  
            preds_ba = occ_pred_ba.reshape(-1, self.background_class)[background_mask]  

            mapping_dict = {v: i for i, v in enumerate(background_class)}  # Background class mapping
            mapped_background_voxels = torch.tensor([mapping_dict[v.item()] for v in valid_background_voxels], device=preds_ba.device)
            if self.class_balance:
                num_total_samples_ba = 0
                for i in range(self.background_class):
                    num_total_samples_ba += (mapped_background_voxels == i).sum() * self.cls_weights_ba[i]
            else:
                num_total_samples_ba = background_mask.sum()

            loss_occ_ba = self.loss_occ(  
                preds_ba,  
                mapped_background_voxels,  
                avg_factor=num_total_samples_ba  
            )  

            loss['loss_occ_tri'] = self.weight_tri_loss*loss_occ_tri
            loss['loss_occ_fo'] = self.weight_foreground_loss*loss_occ_fo
            loss['loss_occ_ba'] = self.weight_background_loss*loss_occ_ba

        return loss  

    def get_occ(self, occ_pred_tri, occ_pred_ba, occ_pred_fo, img_metas=None):  
        """  
        Args:  
            occ_pred_tri: (B, Dx, Dy, Dz, 3) -> Three-class prediction (foreground/background/free)  
            occ_pred_ba: (B, Dx, Dy, Dz, 8) -> Fine-grained background prediction (8 classes)  
            occ_pred_fo: (B, Dx, Dy, Dz, 9) -> Fine-grained foreground prediction (9 classes)  
            img_metas: Metadata, possibly containing location information (optional)  

        Returns:  
            List[np.ndarray]: List of per-sample voxel predictions, shape `(Dx, Dy, Dz)`, dtype `uint8`
        """  
        occ_score_tri = occ_pred_tri.softmax(-1)   # (B, Dx, Dy, Dz, 3) -> Softmax for three-class  
        occ_res_tri = occ_score_tri.argmax(-1)     # (B, Dx, Dy, Dz) -> Three-class result: 0=background, 1=foreground, 2=free space  

        B, Dx, Dy, Dz = occ_res_tri.shape  
        occ_res = torch.full((B, Dx, Dy, Dz), 17, dtype=torch.uint8, device=occ_pred_tri.device)  # Set all to 17 initially  


        background_masks = (occ_res_tri == 0)      # (B, Dx, Dy, Dz)  
        foreground_masks = (occ_res_tri == 1)      # (B, Dx, Dy, Dz)  
        # No need to handle free space since it is already initialized as 17  

        occ_pred_ba_softmax = occ_pred_ba.softmax(-1)          # (B, Dx, Dy, Dz, 8)  
        occ_res_ba = occ_pred_ba_softmax.argmax(-1)            # (B, Dx, Dy, Dz)  
        background_class_tensor = torch.tensor(background_class, device=occ_pred_ba.device, dtype=torch.uint8)  # Background class tensor
        # Map predicted background class indices back to original classes  
        mapped_ba = background_class_tensor[occ_res_ba]               # (B, Dx, Dy, Dz)  
        occ_res[background_masks] = mapped_ba[background_masks]  # Assign background class predictions  

        occ_pred_fo_softmax = occ_pred_fo.softmax(-1)          # (B, Dx, Dy, Dz, 9)  
        occ_res_fo = occ_pred_fo_softmax.argmax(-1)            # (B, Dx, Dy, Dz)  
        foreground_class_tensor = torch.tensor(foreground_class, device=occ_pred_fo.device, dtype=torch.uint8)  # Foreground class tensor
        # Map predicted foreground class indices back to original classes  
        mapped_fo = foreground_class_tensor[occ_res_fo]               # (B, Dx, Dy, Dz)  
        occ_res[foreground_masks] = mapped_fo[foreground_masks]  # Assign foreground class predictions  

        occ_res_np = occ_res.cpu().numpy()                     # Convert to numpy array  
        occ_res_list = [occ_res_np[b] for b in range(B)]       # Convert to list  (B, Dx, Dy, Dz)

        return occ_res_list  




    

@HEADS.register_module()
class BEVOCCHead_FO(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=512,
                 occ_fo_dim = 64,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 num_fo_classes = 9,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ=None,
                 loss_occ_fo=None,
                 loss_occ_final=None,
                 loss_mask_fo = None,
                 loss_occ_weight=1.0,
                 loss_occ_fo_weight=1.0,
                 loss_occ_final_weight=1.0,
                 loss_mask_fo_weight=1.0
                 ):
        super(BEVOCCHead_FO, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        self.occ_fo_dim = occ_fo_dim
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.loss_occ_weight = loss_occ_weight
        self.loss_occ_fo_weight = loss_occ_fo_weight
        self.loss_occ_final_weight = loss_occ_final_weight
        self.loss_mask_fo_weight = loss_mask_fo_weight
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter1 = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, self.occ_fo_dim * Dz),
                nn.Tanh()  
            )
            self.predicter2 = nn.Sequential(
                nn.Linear(self.occ_fo_dim, self.occ_fo_dim * 2),
                nn.Softplus(),
                nn.Linear(self.occ_fo_dim * 2, num_classes),
            )
        self.weights_pred = nn.Sequential(
            nn.Linear(self.occ_fo_dim * 2, self.occ_fo_dim * 2),
            nn.Linear(self.occ_fo_dim * 2, 1),
            nn.Sigmoid(),
        )
        self.occ_fo_predicter = nn.Sequential(
            nn.Linear(self.occ_fo_dim, self.occ_fo_dim*2),
            nn.Softplus(),
            nn.Linear(self.occ_fo_dim * 2, num_fo_classes + 1) 
        )
        self.final_predicter = nn.Sequential(
            nn.Linear(self.occ_fo_dim * 2, self.occ_fo_dim * 2),
            nn.Softplus(),
            nn.Linear(self.occ_fo_dim * 2, num_classes) 
        )

        self.use_mask = use_mask
        self.num_classes = num_classes
        self.num_fo_classes = num_fo_classes
        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
            loss_occ['class_weight'] = class_weights        # ce loss
            loss_occ_final['class_weight'] = class_weights        # ce loss

            remapped_frequencies = np.zeros(num_fo_classes + 1)  

            for new_class, original_class in enumerate(foreground_class):  
                remapped_frequencies[new_class] = nusc_class_frequencies[original_class]  
    
            remaining_classes = set(range(len(nusc_class_frequencies))) - set(foreground_class)  
            remapped_frequencies[9] = nusc_class_frequencies[list(remaining_classes)].sum()  
            
            foreground_withfree_weights = torch.from_numpy(1 / np.log(remapped_frequencies + 0.001))  
           
            self.cls_fo_free_weights = foreground_withfree_weights  
            loss_occ_fo['class_weight'] = self.cls_fo_free_weights  

        self.loss_occ = build_loss(loss_occ)
        self.loss_occ_fo = build_loss(loss_occ_fo)
        self.loss_occ_final = build_loss(loss_occ_final)
        self.loss_mask_fo = build_loss(loss_mask_fo)
        

    def forward(self, img_feats, occ_feats, training_process = True):
        """
        Args:
            img_feats: (B, C, Dy, Dx)
            occ_feats: (B, C, Dz, Dy, Dx)
        Returns:

        """
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, Dz, C) --> (B, Dx, Dy, Dz, n_cls)
            occ_pred = self.predicter1(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.occ_fo_dim)
            occ_extral_feature = occ_pred
            if training_process:
                occ_pred = self.predicter2(occ_pred)
        # (B, C, Dz, Dy, Dx) -> (B, Dx, Dy, Dz, 10)
        occ_feats = occ_feats.permute(0, 4, 3, 2, 1)
        bs_o, Dx_o, Dy_o, Dz_o = occ_feats.shape[:4]
        if training_process:
            occ_fo_pred = self.occ_fo_predicter(occ_feats)
            occ_fo_pred = occ_fo_pred.view(bs_o, Dx_o, Dy_o, Dz_o, 10)
        # gates: (B, Dx, Dy, Dz, 2)
        fo_weights_voxel = self.weights_pred(torch.cat([occ_extral_feature, occ_feats], dim=-1))

        merged_occ_feature = torch.cat([(1-fo_weights_voxel) * occ_extral_feature, fo_weights_voxel * occ_feats], dim=-1)
        occ_final_pred = self.final_predicter(merged_occ_feature)
        if training_process:
            return occ_pred, occ_fo_pred, occ_final_pred
        else:
            return occ_final_pred


    def loss(self, occ_pred, occ_fo_pred, occ_final_pred, voxel_semantics, mask_camera, mask_fo):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            occ_fo_pred: (B, Dx, Dy, Dz, 10)
            occ_final_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
            mask_fo: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            voxel_semantics = voxel_semantics.reshape(-1)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
            preds = occ_pred.reshape(-1, self.num_classes)
            final_pred = occ_final_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, 10)
            occ_fo_pred = occ_fo_pred.reshape(-1, self.num_fo_classes+1)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            mask_camera = mask_camera.reshape(-1)
            mask_fo = mask_fo.reshape(-1)

   
            mapped_foreground_voxels = self.remap_labels(voxel_semantics, foreground_class)
            mapped_foreground_voxels_valid = mapped_foreground_voxels[mask_camera.bool()]
            mapped_mask_fo_voxels = torch.where(mapped_foreground_voxels == 9, 0, 1) 

            loss_mask_fo = self.loss_mask_fo(mask_fo, mapped_mask_fo_voxels, 
                                                    mask_camera, use_mask_camera=True)

            if self.class_balance:
                num_total_samples_fo = 0
                for i in range(self.num_fo_classes + 1):
                    num_total_samples_fo += (mapped_foreground_voxels_valid == i).sum() * self.cls_fo_free_weights[i]
            else:
                num_total_samples_fo = mask_camera.sum()

            loss_occ_fo = self.loss_occ_fo(
                occ_fo_pred,                     
                mapped_foreground_voxels,      
                mask_camera, 
                avg_factor=num_total_samples_fo
            )
            if self.class_balance:
                valid_voxels = voxel_semantics[mask_camera.bool()]
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = mask_camera.sum()

            loss_occ = self.loss_occ(
                preds,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                mask_camera,        # (B*Dx*Dy*Dz, )
                avg_factor=num_total_samples
            )
            loss_occ_final = self.loss_occ_final(
                final_pred,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                mask_camera,        # (B*Dx*Dy*Dz, )
                avg_factor=num_total_samples
            )
            loss['loss_mask_fo'] = loss_mask_fo * self.loss_mask_fo_weight
            loss['loss_occ_fo'] = loss_occ_fo * self.loss_occ_fo_weight
            loss['loss_occ'] = loss_occ * self.loss_occ_weight
            loss['loss_occ_final'] = loss_occ_final * self.loss_occ_final_weight
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)
            final_pred = occ_final_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, 10)
            occ_fo_pred = occ_fo_pred.reshape(-1, self.num_fo_classes+1)

            mask_fo = mask_fo.reshape(-1)
            
     
            mapped_foreground_voxels = self.remap_labels(voxel_semantics, foreground_class) 

            mapped_mask_fo_voxels = torch.where(mapped_foreground_voxels == 9, 0, 1) 
            loss_mask_fo = self.loss_mask_fo(mask_fo, mapped_mask_fo_voxels)
        
            if self.class_balance:
                num_total_samples_fo = 0
                for i in range(self.num_fo_classes + 1):
                    num_total_samples_fo += (mapped_foreground_voxels == i).sum() * self.cls_fo_free_weights[i]
            else:
                num_total_samples_fo = len(mapped_foreground_voxels)

            loss_occ_fo = self.loss_occ_fo(
                occ_fo_pred,                     
                mapped_foreground_voxels,      
                avg_factor=num_total_samples_fo
            )
        
            if self.class_balance:
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = len(voxel_semantics)

            loss_occ = self.loss_occ(
                preds,
                voxel_semantics,
                avg_factor=num_total_samples
            )
            loss_occ_final = self.loss_occ_final(
                final_pred,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                avg_factor=num_total_samples
            )
            loss['loss_mask_fo'] = loss_mask_fo * self.loss_mask_fo_weight
            loss['loss_occ_fo'] = loss_occ_fo * self.loss_occ_fo_weight
            loss['loss_occ'] = loss_occ * self.loss_occ_weight
            loss['loss_occ_final'] = loss_occ_final * self.loss_occ_final_weight
            
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)
    
    def remap_labels(self, voxel_semantics, foreground_class):  

        foreground_class = torch.tensor(foreground_class, device=voxel_semantics.device)  
        mapping = torch.full((18,), 9, dtype=torch.long, device=voxel_semantics.device) 
        

        for new_class, original_class in enumerate(foreground_class):  
            mapping[original_class] = new_class  

        remapped_labels = mapping[voxel_semantics]  
        
        return remapped_labels  



@HEADS.register_module()
class BEVOCCHead2D_V2(BaseModule):      # Use stronger loss setting
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ=None,
                 ):
        super(BEVOCCHead2D_V2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz

        # voxel-level prediction
        self.occ_convs = nn.ModuleList()
        self.final_conv = ConvModule(
            in_dim,
            self.out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes * Dz),
            )

        self.use_mask = use_mask
        self.num_classes = num_classes

        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
        self.loss_occ = build_loss(loss_occ)

    def forward(self, img_feats):
        """
        Args:
            img_feats: (B, C, Dy=200, Dx=200)
            img_feats: [(B, C, 100, 100), (B, C, 50, 50), (B, C, 25, 25)]   if ms
        Returns:

        """
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, 2*C) --> (B, Dx, Dy, Dz*n_cls)
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.num_classes)

        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()    # (B, Dx, Dy, Dz)
        preds = occ_pred.permute(0, 4, 1, 2, 3).contiguous()    # (B, n_cls, Dx, Dy, Dz)
        loss_occ = self.loss_occ(
            preds,
            voxel_semantics,
            weight=self.cls_weights.to(preds),
        ) * 100.0
        loss['loss_occ'] = loss_occ
        loss['loss_voxel_sem_scal'] = sem_scal_loss(preds, voxel_semantics)
        loss['loss_voxel_geo_scal'] = geo_scal_loss(preds, voxel_semantics, non_empty_idx=17)
        loss['loss_voxel_lovasz'] = lovasz_softmax(torch.softmax(preds, dim=1), voxel_semantics)

        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)

    def get_occ_gpu(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1).int()      # (B, Dx, Dy, Dz)
        return list(occ_res)
    


@HEADS.register_module()
class BEVOCCHead_FO_Noheads(BaseModule):
    def __init__(self,
                 in_dim=256,
                 out_dim=512,
                 occ_fo_dim = 64,
                 Dz=16,
                 use_mask=True,
                 num_classes=18,
                 num_fo_classes = 9,
                 use_predicter=True,
                 class_balance=False,
                 loss_occ_final=None,
                 loss_mask_fo = None,
                 loss_occ_final_weight=1.0,
                 loss_mask_fo_weight=1.0
                 ):
        super(BEVOCCHead_FO_Noheads, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Dz = Dz
        self.occ_fo_dim = occ_fo_dim
        out_channels = out_dim if use_predicter else num_classes * Dz
        self.loss_occ_final_weight = loss_occ_final_weight
        self.loss_mask_fo_weight = loss_mask_fo_weight
        self.final_conv = ConvModule(
            self.in_dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter1 = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, self.occ_fo_dim * Dz),
                nn.Tanh()  
            )
        self.weights_pred = nn.Sequential(
            nn.Linear(self.occ_fo_dim * 2, self.occ_fo_dim * 2),
            nn.Linear(self.occ_fo_dim * 2, 1),
            nn.Sigmoid(),
        )
        self.final_predicter = nn.Sequential(
            nn.Linear(self.occ_fo_dim * 2, self.occ_fo_dim * 2),
            nn.Softplus(),
            nn.Linear(self.occ_fo_dim * 2, num_classes) 
        )

        self.use_mask = use_mask
        self.num_classes = num_classes
        self.class_balance = class_balance
        if self.class_balance:
            class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_classes] + 0.001))
            self.cls_weights = class_weights
            loss_occ_final['class_weight'] = class_weights        # ce loss
           
           
        self.loss_occ_final = build_loss(loss_occ_final)
        self.loss_mask_fo = build_loss(loss_mask_fo)
        

    def forward(self, img_feats, occ_feats, training_process = True):
        """
        Args:
            img_feats: (B, C, Dy, Dx)
            occ_feats: (B, C, Dz, Dy, Dx)
        Returns:

        """
        # (B, C, Dy, Dx) --> (B, C, Dy, Dx) --> (B, Dx, Dy, C)
        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, Dx, Dy = occ_pred.shape[:3]
        if self.use_predicter:
            # (B, Dx, Dy, C) --> (B, Dx, Dy, Dz, C) --> (B, Dx, Dy, Dz, n_cls)
            occ_pred = self.predicter1(occ_pred)
            occ_pred = occ_pred.view(bs, Dx, Dy, self.Dz, self.occ_fo_dim)
            occ_extral_feature = occ_pred
        # (B, C, Dz, Dy, Dx) -> (B, Dx, Dy, Dz, 10)
        occ_feats = occ_feats.permute(0, 4, 3, 2, 1)
        # gates: (B, Dx, Dy, Dz, 2)
        fo_weights_voxel = self.weights_pred(torch.cat([occ_extral_feature, occ_feats], dim=-1))
       
        merged_occ_feature = torch.cat([(1-fo_weights_voxel) * occ_extral_feature, fo_weights_voxel * occ_feats], dim=-1)
        occ_final_pred = self.final_predicter(merged_occ_feature)
        if training_process:
            return occ_final_pred
        else:
            return occ_final_pred


    def loss(self, occ_final_pred, voxel_semantics, mask_camera, mask_fo):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
            occ_fo_pred: (B, Dx, Dy, Dz, 10)
            occ_final_pred: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
            mask_fo: (B, Dx, Dy, Dz)
        Returns:

        """
        loss = dict()
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)   # (B, Dx, Dy, Dz)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            voxel_semantics = voxel_semantics.reshape(-1)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, n_cls)
            final_pred = occ_final_pred.reshape(-1, self.num_classes)
            # (B, Dx, Dy, Dz, n_cls) --> (B*Dx*Dy*Dz, 10)
            # (B, Dx, Dy, Dz) --> (B*Dx*Dy*Dz, )
            mask_camera = mask_camera.reshape(-1)
            mask_fo = mask_fo.reshape(-1)

    
            mapped_foreground_voxels = self.remap_labels(voxel_semantics, foreground_class)
            mapped_mask_fo_voxels = torch.where(mapped_foreground_voxels == 9, 0, 1) 

            loss_mask_fo = self.loss_mask_fo(mask_fo, mapped_mask_fo_voxels, 
                                                    mask_camera, use_mask_camera=True)

            if self.class_balance:
                valid_voxels = voxel_semantics[mask_camera.bool()]
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (valid_voxels == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = mask_camera.sum()

            loss_occ_final = self.loss_occ_final(
                final_pred,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                mask_camera,        # (B*Dx*Dy*Dz, )
                avg_factor=num_total_samples
            )
            loss['loss_mask_fo'] = loss_mask_fo * self.loss_mask_fo_weight
            loss['loss_occ_final'] = loss_occ_final * self.loss_occ_final_weight
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            final_pred = occ_final_pred.reshape(-1, self.num_classes)

            mask_fo = mask_fo.reshape(-1)
            

            mapped_foreground_voxels = self.remap_labels(voxel_semantics, foreground_class) 

            mapped_mask_fo_voxels = torch.where(mapped_foreground_voxels == 9, 0, 1) 
            loss_mask_fo = self.loss_mask_fo(mask_fo, mapped_mask_fo_voxels)
       
            if self.class_balance:
                num_total_samples = 0
                for i in range(self.num_classes):
                    num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]
            else:
                num_total_samples = len(voxel_semantics)

            loss_occ_final = self.loss_occ_final(
                final_pred,      # (B*Dx*Dy*Dz, n_cls)
                voxel_semantics,    # (B*Dx*Dy*Dz, )
                avg_factor=num_total_samples
            )
            loss['loss_mask_fo'] = loss_mask_fo * self.loss_mask_fo_weight
            loss['loss_occ_final'] = loss_occ_final * self.loss_occ_final_weight
            
        return loss

    def get_occ(self, occ_pred, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, C)
            img_metas:

        Returns:
            List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)    # (B, Dx, Dy, Dz, C)
        occ_res = occ_score.argmax(-1)      # (B, Dx, Dy, Dz)
        occ_res = occ_res.cpu().numpy().astype(np.uint8)     # (B, Dx, Dy, Dz)
        return list(occ_res)
    
    def remap_labels(self, voxel_semantics, foreground_class):  
 
        foreground_class = torch.tensor(foreground_class, device=voxel_semantics.device)  
        mapping = torch.full((18,), 9, dtype=torch.long, device=voxel_semantics.device)
        
       
        for new_class, original_class in enumerate(foreground_class):  
            mapping[original_class] = new_class  
       
        remapped_labels = mapping[voxel_semantics]  
        
        return remapped_labels  