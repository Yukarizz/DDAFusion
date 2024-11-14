from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction,Restormer_Decoder_prompt_pool
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
import open_clip
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

norm = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path=r"models/final_submit.pth"
clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained='models/epoch_30.pt')
clip_model = clip_model.cuda()

for dataset_name in ["MSRS_bad_weather_val_new"]:
    print("\n"*2+"="*80)
    model_name="DDAFusion    "
    # rainy, foggy, snowy, raindrop
    deg_type = "snowy"
    deg_degree = ["lightly","moderately","heavily"]
    print("The test result of "+dataset_name+' :')
    test_folder=os.path.join('test_img',dataset_name,deg_type)
    test_out_folder=os.path.join('test_result',dataset_name,deg_type)
    if not os.path.exists(test_out_folder):
        os.mkdir(test_out_folder)
        os.mkdir(os.path.join(test_out_folder, "lightly"))
        os.mkdir(os.path.join(test_out_folder, "moderately"))
        os.mkdir(os.path.join(test_out_folder, "heavily"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder_prompt_pool()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

    Encoder.load_state_dict(torch.load(ckpt_path)['DIDF_Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()
    ir_paths = os.listdir(os.path.join('test_img',dataset_name,'ir'))
    normal_paths = os.listdir(os.path.join('test_img', dataset_name, 'vi'))
    statistical_prompt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for degree in deg_degree:
        vi_paths = os.listdir(os.path.join('test_img',dataset_name,deg_type,degree))
        with torch.no_grad():
            tqdms = tqdm(enumerate(ir_paths))
            for i,img_name in tqdms:
                tqdms.set_description("Generating %s_%s:" % (deg_type, degree))
                data_IR=image_read_cv2(os.path.join('test_img',dataset_name,"ir",img_name),mode='RGB')[np.newaxis, ...]/255.0
                data_VIS = image_read_cv2(os.path.join(test_folder,degree,vi_paths[i]), mode='RGB')[np.newaxis, ...]/255.0

                data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
                data_VIS, data_IR = data_VIS.cuda().permute(0,3,1,2), data_IR.cuda().permute(0,3,1,2)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    img4clip = image_read_cv2(os.path.join(test_folder,degree,vi_paths[i]), mode='RGB') / 255.0
                    img4clip = clip_transform(img4clip).cuda().unsqueeze(0)
                    image_context, degra_context = clip_model.encode_image(img4clip, control=True)
                    image_context = image_context.float()
                    degra_context = degra_context.float()

                feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
                feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
                feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
                feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
                data_Fuse, data_Fuse_fine, prompt_indices = Decoder(feature_F_B, feature_F_D,degra_context=degra_context,return_index=True)
                for indice in prompt_indices[0]:
                    statistical_prompt[indice.item()] += 1
                fi = np.squeeze((data_Fuse_fine * 255).cpu().permute(0,2,3,1).numpy())
                img_save(fi, img_name.split(sep='.')[0], os.path.join(test_out_folder,degree))

    print(statistical_prompt)
    for degree in deg_degree:
        # print("Evaluating %s_%s:" % (deg_type, degree))
        vi_paths = os.listdir(os.path.join('test_img', dataset_name, deg_type, degree))
        eval_folder = os.path.join(test_out_folder,degree)
        ori_img_folder = os.path.join('test_img',dataset_name)

        metric_result = np.zeros((9))
        tqdms = tqdm(enumerate(ir_paths))
        for i,img_name in tqdms:
            tqdms.set_description("Evaluating %s_%s:" % (deg_type, degree))
            ir = image_read_cv2(os.path.join(ori_img_folder,"ir", img_name), 'GRAY')
            vi = image_read_cv2(os.path.join(ori_img_folder,"vi", normal_paths[i]), 'GRAY')
            fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
            metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                        , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                        , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                        , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)
                                        , Evaluator.PSNR(fi,ir,vi)])

        metric_result /= len(os.listdir(eval_folder))
        print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM\tPSNR")
        print(model_name+': '+deg_type+'_'+degree+'\t'+str(np.round(metric_result[0], 2))+'\t'
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
