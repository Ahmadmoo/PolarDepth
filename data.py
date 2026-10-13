import glob
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image

cv2.setNumThreads(0)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
POL_LAYOUT = (0, 45, 90, 135)  # polarizer angle of the (top-left, top-right, bottom-left, bottom-right) block in HAMMER pol/*.png


def to_tensor(rgb):
    return torch.from_numpy(((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())


def stokes(i0, i45, i90, i135):
    s0 = 0.5 * (i0 + i45 + i90 + i135)
    p = np.stack([i0 - i90, i45 - i135], -1) / (s0[..., None] + 1e-6)
    p /= np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1)
    p[s0 < 0.02] = 0
    return p.astype(np.float32)


def read_hammer_pol(path):
    im = cv2.imread(path, -1)
    im = im.astype(np.float32) / np.iinfo(im.dtype).max
    im = im[..., :3].mean(-1) if im.ndim == 3 else im
    h, w = im.shape[0] // 2, im.shape[1] // 2
    q = dict(zip(POL_LAYOUT, [im[:h, :w], im[:h, w:], im[h:, :w], im[h:, w:]]))
    return stokes(q[0], q[45], q[90], q[135])


class Hammer(torch.utils.data.Dataset):
    def __init__(self, root, scenes, size=(480, 640), train=False):
        self.files = sorted(f for f in glob.glob(os.path.join(root, 'scene*_traj*', 'polarization', 'rgb', '*.png'))
                            if 'naked' not in f and int(f.split(os.sep)[-4][5:].split('_')[0]) in scenes)
        self.size, self.train = size, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        d, n, (h, w) = os.path.dirname(os.path.dirname(f)), os.path.basename(f), self.size
        rgb = cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA)
        pol = cv2.resize(read_hammer_pol(os.path.join(d, 'pol', n)), (w, h), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(cv2.imread(os.path.join(d, '_gt', n), -1).astype(np.float32) / 1000, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.train:
            if random.random() < 0.5:
                rgb, pol, depth = rgb[:, ::-1], pol[:, ::-1] * np.array([1, -1], np.float32), depth[:, ::-1]  # mirror flips AoLP sign -> s2 sign
            rgb = np.clip(rgb * random.uniform(0.8, 1.2) + random.uniform(-20, 20), 0, 255)
        return to_tensor(rgb), torch.from_numpy(pol.transpose(2, 0, 1).copy()), torch.from_numpy(depth.copy())[None]


def booster_files(root):
    return sorted(glob.glob(os.path.join(root, '*', 'camera_00', 'im*.png')))


def read_booster(f):
    s = os.path.dirname(os.path.dirname(f))
    rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    disp = np.load(os.path.join(s, 'disp_00.npy')).astype(np.float32)
    mc = os.path.join(s, 'mask_cat.png')
    cat = np.array(Image.open(mc)) if os.path.exists(mc) else None
    cat = cat[..., 0] if cat is not None and cat.ndim == 3 else cat
    fs = cv2.FileStorage(os.path.join(s, 'calib_00-02.xml'), cv2.FILE_STORAGE_READ)
    return rgb, disp, cat, fs.getNode('proj_matL').mat()[0, 0], fs.getNode('baselineLR').real()
