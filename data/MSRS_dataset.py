import os
import random
import sys

from PIL import Image
import cv2
# import lmdb
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
distortion = ['foggy', 'rainy', 'snowy','raindrop']
try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass

def clip_transform(np_image, resolution=224):
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return Compose([
        Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(resolution),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])(pil_image)

class MSRS_dataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.size = 224
        self.deg_types = distortion

        self.distortion = {}
        GT_paths = util.get_image_paths(
            'img', os.path.join(opt.dataroot,opt.dataset, 'vi')
        )  # GT list

        for deg_type in distortion:
            LR_paths = util.get_image_paths(
                'img', os.path.join(opt.dataroot,opt.dataset, deg_type)
            )  # LR list
            IR_paths = util.get_image_paths(
                'img', os.path.join(opt.dataroot,opt.dataset, 'ir')
            )  # LR list
            self.distortion[deg_type] = (GT_paths+GT_paths+GT_paths, LR_paths,IR_paths+IR_paths+IR_paths)
        deg_type = 'common'
        LR_paths = util.get_image_paths(
                'img', os.path.join(opt.dataroot,opt.dataset, 'vi')
            )
        IR_paths = util.get_image_paths(
            'img', os.path.join(opt.dataroot, opt.dataset, 'ir')
        )  # LR list
        self.distortion[deg_type] = (GT_paths + GT_paths + GT_paths, LR_paths + LR_paths + LR_paths,IR_paths+IR_paths+IR_paths)
        self.data_lens = [len(self.distortion[deg_type][0]) for deg_type in self.deg_types]

        self.random_scale_list = [1]

    def __getitem__(self, index):

        # choose degradation type and data index
        # type_id, deg_type = 0, self.deg_types[0]
        #  while index >= len(self.distortion[deg_type][0]):
        #     type_id += 1
        #     index -= len(self.distortion[deg_type][0])
        #     deg_type = self.deg_types[type_id]

        type_id = int(index % len(self.deg_types))
        if self.opt.phase == "train":
            deg_type = self.deg_types[type_id]
            index = np.random.randint(self.data_lens[type_id])
        else:
            while index // len(self.deg_types) >= self.data_lens[type_id]:
                index += 1
                type_id = int(index % len(self.deg_types))
            deg_type = self.deg_types[type_id]
            index = index // len(self.deg_types)

        # get GT image
        GT_path = self.distortion[deg_type][0][index]
        img_GT = util.read_img(
            None, GT_path, None
        )  # return: Numpy float32, HWC, BGR, [0,1]

        # get LQ image
        LQ_path = self.distortion[deg_type][1][index]
        img_LQ = util.read_img(
            None, LQ_path, None
        )  # return: Numpy float32, HWC, BGR, [0,1]

        # get LQ image
        IR_path = self.distortion[deg_type][2][index]
        img_IR = util.read_img(
            None, IR_path, None
        )  # return: Numpy float32, HWC, BGR, [0,1]

        if self.opt.phase == "train":
            H, W, C = img_GT.shape

            rnd_h = random.randint(0, max(0, H - self.size))
            rnd_w = random.randint(0, max(0, W - self.size))
            img_GT = img_GT[rnd_h : rnd_h + self.size, rnd_w : rnd_w + self.size, :]
            img_LQ = img_LQ[rnd_h : rnd_h + self.size, rnd_w : rnd_w + self.size, :]
            img_IR = img_IR[rnd_h : rnd_h + self.size, rnd_w : rnd_w + self.size, :]
            # augmentation - flip, rotate
            img_LQ, img_GT,img_IR = util.augment(
                [img_LQ, img_GT, img_IR],
                True,
                True,
                mode='Fusion',
            )
        # change color space if necessary
        img_GT = util.channel_convert(img_GT.shape[2], "RGB", [img_GT])[0]
        img_LQ = util.channel_convert(img_LQ.shape[2], "RGB", [img_LQ])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]

        # gt4clip = clip_transform(img_GT)
        lq4clip = clip_transform(img_LQ)
        img_IR = np.repeat(img_IR,3,2)

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        img_IR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_IR, (2, 0, 1)))).float()
        lq4clip = torch.from_numpy(np.ascontiguousarray(lq4clip)).float()
        return {'LQ': img_LQ, 'GT': img_GT, 'IR': img_IR, 'LQ_clip': lq4clip}

    def __len__(self):
        return np.sum(self.data_lens)

