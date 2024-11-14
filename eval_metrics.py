import os
import numpy as np
from utils.Evaluator import Evaluator
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from tqdm import tqdm
def clip_transform(np_image, resolution=224):
    pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
    return Compose([
        Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(resolution),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])(pil_image)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path=r"models/CDDFuse_degra.pth"

for dataset_name in ["MSRS_bad_weather_val_new"]:
    deg_types = ["snowy","foggy","rainy","raindrop"]
    deg_degree = ["heavily","moderately","lightly"]
    ir_paths = os.listdir(r"F:\pythonproject\CDDFuse_paper\test_img\MSRS_bad_weather_val_new\ir")
    test_out_folder = r"F:\pythonproject\Histoformer-main\Allweather\result\MSRS-WD"
    for deg_type in deg_types:
        for degree in deg_degree:
            eval_folder = os.path.join(test_out_folder,deg_type,degree)
            ori_img_folder = os.path.join('test_img',dataset_name)
            vi_names = os.listdir(os.path.join(ori_img_folder,'vi'))
            metric_result = np.zeros((9))
            tqdms = tqdm(enumerate(ir_paths))
            for i,img_name in tqdms:
                tqdms.set_description("Evaluating %s_%s:" % (deg_type, degree))
                ir = image_read_cv2(os.path.join(ori_img_folder,"ir", img_name), 'GRAY')
                vi = image_read_cv2(os.path.join(ori_img_folder,"vi", vi_names[i]), 'GRAY')
                fi = image_read_cv2(os.path.join(eval_folder, vi_names[i]), 'GRAY')

                fi = Image.fromarray(fi).resize((vi.shape[1],vi.shape[0]))
                fi = np.round(np.array(fi))
                metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                            , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                            , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                            , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)
                                            , Evaluator.PSNR(fi,ir,vi)])

            metric_result /= len(os.listdir(eval_folder))
            print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM\tPSNR")
            print(': '+deg_type+'_'+degree+'\t'+str(np.round(metric_result[0], 2))+'\t'
                    +str(np.round(metric_result[1], 2))+'\t'
                    +str(np.round(metric_result[2], 2))+'\t'
                    +str(np.round(metric_result[3], 2))+'\t'
                    +str(np.round(metric_result[4], 2))+'\t'
                    +str(np.round(metric_result[5], 2))+'\t'
                    +str(np.round(metric_result[6], 2))+'\t'
                    +str(np.round(metric_result[7], 2))+'\t'
                    +str(np.round(metric_result[8], 2)) + '\t'
                    )
            print("="*80)