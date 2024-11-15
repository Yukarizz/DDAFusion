import torch
import cv2 as cv
from torchvision.transforms import transforms,ToPILImage,ToTensor
from PIL import Image,ImageOps
import torchvision
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.nn import functional as F
import math


result_path = r"G:\DDAFusion\More\RoadScene\LRRNet\snowy"
a_path = r"F:\pythonproject\Histoformer-main\Allweather\result\RoadScene\snowy"
b_path = r"F:\pythonproject\Histoformer-main\Allweather\result\RoadScene\ir"
save_path = r"G:\DDAFusion\More\RoadScene\LRRNet\color"
def get_imgs_path(path):
    imglist1 = []
    filenames1 = os.listdir(path)
    # if "MFNet" in path:
    #     filenames1.sort(key=lambda x: int(x[:-5]))
    # else:
    #     filenames1.sort(key=lambda x: int(x[5:-4]))
    # index = 0
    # for name in filenames1:
    #     name = name.replace("png","jpg")
    #     filenames1[index] = name
    #     index +=1
    return filenames1


result_imgs_path = get_imgs_path(result_path)
a_imgs_path = get_imgs_path(a_path)
b_imgs_path = get_imgs_path(b_path)
set1 = set(result_imgs_path)
set2 = set(a_imgs_path)

paths = list(set1 and set2)
index = 0

for name in paths:
    y = result_path + "\%s"%name
    result_img = Image.open(y).convert("L")
    w, h = result_img.size
    result_img = ToTensor()(result_img).unsqueeze(0)

    a = a_path + "\%s"%name
    a_img = Image.open(a).convert("YCbCr")
    Ya,Cba,Cra = a_img.split()
    a_img = ToTensor()(a_img)
    c,w,h = a_img.size()
    b = b_path + "\%s" % name
    b_img = Image.open(b.replace("png","jpg").replace('RGB','T')).convert("YCbCr")
    Yb,Cbb,Crb = b_img.split()
    b_img = ToTensor()(b_img)

    result_img = F.interpolate(result_img, size=[w, h])

    Cb_vi = transforms.ToTensor()(Cba).unsqueeze(0)
    Cb_ir = transforms.ToTensor()(Cbb).unsqueeze(0)
    Cr_vi = transforms.ToTensor()(Cra).unsqueeze(0)
    Cr_ir = transforms.ToTensor()(Crb).unsqueeze(0)
    Cb_fused = (Cb_vi * (torch.abs(Cb_vi - 128)) + Cb_ir * (torch.abs(Cb_ir - 128))) / (
                torch.abs(Cb_vi - 128) + torch.abs(Cb_ir - 128))
    Cr_fused = (Cr_vi * (torch.abs(Cr_vi - 128)) + Cr_ir * (torch.abs(Cr_ir - 128))) / (
                torch.abs(Cr_vi - 128) + torch.abs(Cr_ir - 128))
    Cb = Cb_vi.squeeze(0)
    Cb = ToPILImage()(Cb)
    Cr = Cr_vi.squeeze(0)
    Cr = ToPILImage()(Cr)
    result_img = ToPILImage()(result_img.squeeze(0))
    result_img = Image.merge("YCbCr", (result_img, Cb, Cr))
    final_save_path = os.path.join(save_path + "/" +'%s' % name)
    result_img = result_img.convert("RGB")
    result_img.save(final_save_path)
    index+=1
    print(index)


