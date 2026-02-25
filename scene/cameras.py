
import os
import pickle
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,gt_mask,sentence,category,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.category=category
        self.image_name = image_name
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        self.gt_mask={k:mask.clamp(0.0, 1.0) for k,mask in gt_mask.items()}
        #self.mask={k:mask for k,mask in gt_mask.items()}
        self.sentence=sentence
        self.original_image = image.clamp(0.0, 1.0)
        
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))
            
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self._dinov2_cache = {}
    def get_language_feature(self, language_feature_dir, feature_level):
        language_feature_name = os.path.join(language_feature_dir, self.image_name)
        seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy'))
        feature_map = torch.from_numpy(np.load(language_feature_name + '_f.npy'))
        
        y, x = torch.meshgrid(torch.arange(0, self.image_height), torch.arange(0, self.image_width))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        seg = seg_map[:, y, x].squeeze(-1).long()
        mask = seg != -1
        if feature_level == 0: # default
            point_feature1 = feature_map[seg[0:1]].squeeze(0)
            mask = mask[0:1].reshape(1, self.image_height, self.image_width)
        elif feature_level == 1: # s
            point_feature1 = feature_map[seg[1:2]].squeeze(0)
            mask = mask[1:2].reshape(1, self.image_height, self.image_width)
        elif feature_level == 2: # m
            point_feature1 = feature_map[seg[2:3]].squeeze(0)
            mask = mask[2:3].reshape(1, self.image_height, self.image_width)
        elif feature_level == 3: # l
            point_feature1 = feature_map[seg[3:4]].squeeze(0)
            mask = mask[3:4].reshape(1, self.image_height, self.image_width)
        else:
            raise ValueError("feature_level=", feature_level)
        point_feature = point_feature1.reshape(self.image_height, self.image_width, -1).permute(2, 0, 1)
       
        return point_feature.cuda(), mask.cuda()

    def get_dinov2_feature(self, feature_dir, expected_dim=None, target_hw=None):
        if feature_dir is None:
            raise ValueError("dinov2 feature_dir is None")
        if target_hw is None:
            target_hw = (self.image_height, self.image_width)

        cache_key = (feature_dir, target_hw, expected_dim)
        if cache_key in self._dinov2_cache:
            return self._dinov2_cache[cache_key]

        npz_path = os.path.join(feature_dir, self.image_name + ".npz")
        npy_path = os.path.join(feature_dir, self.image_name + ".npy")
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            feat = data["feat"]
        elif os.path.exists(npy_path):
            feat = np.load(npy_path)
        else:
            raise FileNotFoundError(f"Missing DINOv2 feature for {self.image_name}")

        if feat.ndim != 3:
            raise ValueError(f"Unexpected DINOv2 feature shape: {feat.shape}")

        if expected_dim is not None and feat.shape[0] != expected_dim:
            if feat.shape[-1] == expected_dim:
                feat = feat.transpose(2, 0, 1)
            else:
                raise ValueError(
                    f"DINOv2 feature dim mismatch: got {feat.shape}, expected {expected_dim}"
                )

        feat_tensor = torch.from_numpy(feat).float().unsqueeze(0)
        if feat_tensor.shape[-2:] != target_hw:
            feat_tensor = F.interpolate(
                feat_tensor, size=target_hw, mode="bilinear", align_corners=False
            )

        feat_tensor = feat_tensor.squeeze(0).to(self.data_device)
        self._dinov2_cache[cache_key] = feat_tensor
        return feat_tensor

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

