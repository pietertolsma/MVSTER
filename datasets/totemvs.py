from torch.utils.data import Dataset
from datasets.data_io import *
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T
import random
from scipy.spatial.transform import Rotation as R
import json
from skimage.transform import resize

from torch import Tensor

def check_invalid_input(imgs, depths, masks, depth_mins, depth_maxs):
    for img in imgs:
        assert np.isnan(img).sum() == 0
        assert np.isinf(img).sum() == 0
    for depth in depths.values():
        assert np.isnan(depth).sum() == 0
        assert np.isinf(depth).sum() == 0
    for mask in masks.values():
        assert np.isnan(mask).sum() == 0
        assert np.isinf(mask).sum() == 0

    assert (depth_mins<=0) == 0
    assert (depth_maxs<=depth_mins) == 0


class MVSDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        super(MVSDataset, self).__init__()
        assert mode in ["train", "val", "test"]
        self.levels = cfg.fpn.levels
        self.datapath = cfg.data.root
        self.material = cfg.data.material
        # self.split = split
        self.listfile = os.path.join(f"./lists/ToteMVS/{mode}.txt")
        self.robust_train = cfg.train.robust
        # assert self.split in ['train', 'val', 'all'], \
        #     'split must be either "train", "val" or "all"!'

        self.img_wh = cfg.data.img_wh
        if self.img_wh is not None:
            assert self.img_wh[0]%32==0 and self.img_wh[1]%32==0, \
                'img_wh must both be multiples of 32!'
        self.nviews = cfg.data.nviews
        self.scale_factors = {} # depth scale factors for each scan
        self.scale_factor = 0 # depth scale factors for each scan
        self.build_metas()

        self.cams = self.read_cam_file(f"{self.datapath}/cams.json")
        self.depth_min = cfg.data.depth_min
        self.depth_max = cfg.data.depth_max

        self.color_augment = T.ColorJitter(brightness=0.5, contrast=0.5)

    def build_metas(self):
        self.metas = []
        with open(self.listfile) as f:
            self.scans = [line.rstrip() for line in f.readlines()]

        for scan in self.scans:

            ref_views = np.arange(0, self.nviews)
            all_src_views = [np.delete(ref_views, i, None) for i in ref_views]

            for (ref, src_views) in zip(ref_views, all_src_views):
                self.metas += [(scan, ref, list(src_views))]

    def read_cam_file(self, filename):
        return json.load(open(filename))

    def read_depth_mask(self, scan, filename, depth_min, depth_max, scale):
        w, h = self.img_wh[0], self.img_wh[1]
        depth = np.load(filename)

        # depth = (depth * self.scale_factor) * scale
        if scan not in self.scale_factors:
            self.scale_factors[scan] = 100.0 / depth_min
        depth = (depth * self.scale_factors[scan]) * scale
        # TODO: Fix this part -> scale factors is weird.
        # depth = depth * scale
        depth = resize(depth, self.img_wh, anti_aliasing=True)
        depth = depth[:, :, None]

        mask = (depth>=depth_min) & (depth<=depth_max)
        assert mask.sum() > 0
        mask = mask.astype(np.float32)
        if self.img_wh is not None:
            depth = cv2.resize(depth, self.img_wh,
                                 interpolation=cv2.INTER_NEAREST)
        h, w = depth.shape
        depth_ms = {}
        mask_ms = {}

        for i, scale in enumerate(self.levels):
            depth_cur = cv2.resize(depth, (w//(2**scale), h//(2**scale)), interpolation=cv2.INTER_NEAREST)
            mask_cur = cv2.resize(mask, (w//(2**scale), h//(2**scale)), interpolation=cv2.INTER_NEAREST)

            depth_ms[f"stage{len(self.levels)-i}"] = depth_cur
            mask_ms[f"stage{len(self.levels)-i}"] = mask_cur

        return depth_ms, mask_ms


    def read_img(self, filename):
        img = Image.open(filename)
        img = img.convert("RGB").resize(self.img_wh)
        # img = self.color_augment(img)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        
        if self.robust_train:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.nviews - 1)
            view_ids = [ref_view] + [src_views[i] for i in index]
            scale = random.uniform(0.8, 1.25)

        else:
            view_ids = [ref_view] + src_views[:self.nviews - 1]
            scale = 1

        imgs = []
        mask = None
        depth = None
        depth_min = None
        depth_max = None

        proj={}

        proj_matrices = [[] for _ in range(len(self.levels))]
 

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, 
                            f"{scan}/{self.material}/{vid}.png")
            depth_filename = os.path.join(self.datapath, 
                        f"{scan}/depth/{vid}.npy")
            #proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            img = self.read_img(img_filename)
            imgs.append(img.transpose(2,0,1))

            #intrinsics, extrinsics, depth_min_, depth_max_ = self.read_cam_file(scan, proj_mat_filename)
            # proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)
            extrinsics = self.cams[str(vid)]["extrinsic"]
            intrinsics = self.cams[str(vid)]["intrinsic"]
            depth_min_, depth_max_ = self.depth_min, self.depth_max

            for j, level in enumerate(self.levels):
                scale_index = len(self.levels) - j - 1
                proj_matrices[scale_index].append(np.zeros(shape=(2, 4, 4), dtype=np.float32))
                proj_matrices[scale_index][-1][0, :4, :4] = extrinsics.copy()
                proj_matrices[scale_index][-1][1, :3, :3] = intrinsics.copy()
                proj_matrices[scale_index][-1][0, :3, 3] /=  (scale * 2**level)
                proj_matrices[scale_index][-1][1, :2, :] /=  (scale * 2**level)

            if i == 0:  # reference view
                depth_min = depth_min_ * scale
                depth_max = depth_max_ * scale
                depth, mask = self.read_depth_mask(scan, depth_filename, depth_min, depth_max, scale)
                for l in range(len(self.levels)):
                    mask[f'stage{l+1}'] = mask[f'stage{l+1}'] # np.expand_dims(mask[f'stage{l+1}'],2)
                    depth[f'stage{l+1}'] = depth[f'stage{l+1}']

        for i in range(len(self.levels)):
            proj[f"stage{i+1}"] = np.stack(proj_matrices[i])

        # check_invalid_input(imgs, depth, mask, depth_min, depth_max)
        # data is numpy array
        return {"imgs": imgs,                   # [Nv, 3, H, W]
                "proj_matrices": proj,          # [N,2,4,4]
                "depth": depth,                 # [1, H, W]
                "depth_values": np.array([depth_min, depth_max], dtype=np.float32),
                "mask": mask,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}                   # [1, H, W]
        