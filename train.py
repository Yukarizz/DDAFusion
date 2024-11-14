# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from net import Restormer_Encoder, Restormer_Decoder_prompt_pool, BaseFeatureExtraction, DetailFeatureExtraction, Restormer_Decoder
from utils.dataset import H5Dataset
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc, clip_degra_loss
import kornia
import open_clip
from data import create_dataloader, create_dataset
from data.MSRS_dataset import MSRS_dataset
from data.data_sampler import DistIterSampler
import argparse
from torchvision.utils import save_image,make_grid
from torchvision.transforms import Normalize
'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''
#### setup options of three networks
parser = argparse.ArgumentParser()
parser.add_argument("-daclip_path", type=str, help="Path to pretrained daclip checkpoint.")
parser.add_argument("-dataroot", type=str, help="Dataset root to the training data.")
parser.add_argument("-dataset", default="MSRS_bad_weather_new", type=str, help="Dataset root to the training data.")
parser.add_argument("-distortion", default=['foggy', 'rainy', 'snowy','raindrop'], type=str,
                    help="Dataset root to the training data.")
parser.add_argument("-phase", default='train', type=str, help="Train or val")
parser.add_argument("-color", default='RGB', type=str, help="Which color mode to use.")
parser.add_argument("-save_frequency", default=5, type=str, help="Frequency to save checkpoint.")
parser.add_argument("-show_frequency", default=1000, type=str, help="Frequency to save checkpoint.")
parser.add_argument("-resume", default=None, type=str, help="Frequency to save checkpoint.")
# parser.add_argument("-epoch_gap", default=0, type=str, help="epoches of Phase I")
parser.add_argument(
    "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
)
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()

# convert to NoneDict, which returns None for missing keys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

clip_model, preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=args.daclip_path)
for param in clip_model.parameters():
    param.requires_grad = False
clip_transform = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
tokenizer = open_clip.get_tokenizer('ViT-B-32')
degradations = ['foggy lightly','foggy moderately','foggy heavily','rainy lightly',
                'rainy moderately','rainy heavily','snowy lightly','snowy moderately','snowy heavily',
                'raindrop lightly','raindrop moderately','raindrop heavily','common']
text_degradations = tokenizer(degradations)
with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = clip_model.encode_text(text_degradations)
clip_model = clip_model.cuda()
# tokenizer = open_clip.get_tokenizer('ViT-B-32')
criteria_fusion = Fusionloss()


# . Set the hyper-parameters for training
num_epochs = 30  # total epoch
# epoch_gap = 1  # epoches of Phase I

lr = 1e-4
weight_decay = 0
batch_size = 2
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1.  # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.  # alpha2 and alpha4
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder_prompt_pool()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
if args.resume:
    print("Resume checkpoint from %s" % args.resume)
    state_dict = torch.load(args.resume)
    DIDF_Encoder.load_state_dict(state_dict['DIDF_Encoder'])
    DIDF_Decoder.load_state_dict(state_dict['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(state_dict['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(state_dict['DetailFuseLayer'])
# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')

# data loader
trainset = MSRS_dataset(args)
trainloader = DataLoader(trainset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()
train_phase = "decom"
for epoch in range(num_epochs):
    ''' train '''
    for i, traindata in enumerate(loader['train']):
        data_VIS, data_IR = traindata['LQ'].cuda(), traindata['IR'].cuda()
        data_HQ, img4clip = traindata['GT'].cuda(), traindata['LQ_clip'].cuda()
        '''get content context and degradation context'''
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_context, degra_context = clip_model.encode_image(img4clip, control=True)
            image_context = image_context.float()
            degra_context = degra_context.float()
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        # if epoch < args.epoch_gap:  # Phase I
        #     train_phase = "decom"
        #     feature_V_B, feature_V_D, _ = DIDF_Encoder(data_HQ)
        #     feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
        #     data_VIS_hat, _ = DIDF_Decoder(None, feature_V_B, feature_V_D)
        #     data_IR_hat, _ = DIDF_Decoder(None, feature_I_B, feature_I_D)
        #
        #     cc_loss_B = cc(feature_V_B, feature_I_B)
        #     cc_loss_D = cc(feature_V_D, feature_I_D)
        #     mse_loss_V = 5 * Loss_ssim(data_HQ, data_VIS_hat) + MSELoss(data_HQ, data_VIS_hat)
        #     mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)
        #
        #     Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_HQ),
        #                            kornia.filters.SpatialGradient()(data_VIS_hat))
        #
        #     loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
        #
        #     loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
        #            mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss
        #
        #     loss.backward()
        #     nn.utils.clip_grad_norm_(
        #         DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        #     nn.utils.clip_grad_norm_(
        #         DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        #     optimizer1.step()
        #     optimizer2.step()
        # else:  # Phase II
        train_phase = "fuse"
        feature_V_B, feature_V_D, feature_V = DIDF_Encoder(data_VIS)
        feature_I_B, feature_I_D, feature_I = DIDF_Encoder(data_IR)
        feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
        feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)
        data_Fuse, data_Fuse_fine = DIDF_Decoder(base_feature=feature_F_B, detail_feature=feature_F_D,
                                            degra_context=degra_context)

        data_Fuse_fine_clip = torch.nn.functional.interpolate(data_Fuse_fine,224)
        data_Fuse_fine_clip = clip_transform(data_Fuse_fine_clip)
        with torch.no_grad(), torch.cuda.amp.autocast():
            fimage_context, fdegra_context = clip_model.encode_image(data_Fuse_fine_clip, control=True)
            fimage_context = fimage_context.float()
            fdegra_context = fdegra_context.float()
        # mse_loss_V = 5*Loss_ssim(data_HQ, data_Fuse) + MSELoss(data_HQ, data_Fuse)
        # mse_loss_I = 5*Loss_ssim(data_IR,  data_Fuse) + MSELoss(data_IR,  data_Fuse)

        cc_loss_B = cc(feature_V_B, feature_I_B)
        cc_loss_D = cc(feature_V_D, feature_I_D)
        loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
        fusionloss, loss_in, loss_grad,loss_color = criteria_fusion(data_HQ, data_IR, data_Fuse_fine,data_Fuse.detach())
        clip_loss, degra_type = clip_degra_loss(text_features,fdegra_context)
        clip_loss = 0.3*clip_loss
        loss = fusionloss + coeff_decomp * loss_decomp + clip_loss
        loss.backward()
        nn.utils.clip_grad_norm_(
            DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()

        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f][loss_in: %f][loss_grad: %f][loss_color: %f][clip: %f][%s] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                loss_in.item(),
                loss_grad.item(),
                loss_color.item(),
                clip_loss.item(),
                degra_type,
                time_left,
            )
        )
        if i % args.show_frequency == 0:
            save_image(data_VIS[0], "train_output/fusion/ori_epoch%s_%sth.png" % (str(epoch), i))
            save_image(data_Fuse[0], "train_output/fusion/fuse_epoch%s_%sth.png" % (str(epoch), i))
            # grid = torch.cat([data_VIS, data_IR, data_Fuse_fine], dim=0)
            # grid = make_grid(grid, nrow=2)
            # save_image(grid, "train_output/fusion/fuse_%s_epoch%s_%sth.png" % (degra_type,str(epoch), i))
    # adjust the learning rate

    scheduler1.step()
    scheduler2.step()
    if not epoch < args.epoch_gap:
        scheduler3.step()
        scheduler4.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6

    if epoch % args.save_frequency == 0:
        checkpoint = {
            'DIDF_Encoder': DIDF_Encoder.state_dict(),
            'DIDF_Decoder': DIDF_Decoder.state_dict(),
            'BaseFuseLayer': BaseFuseLayer.state_dict(),
            'DetailFuseLayer': DetailFuseLayer.state_dict(),
        }
        torch.save(checkpoint, os.path.join("models/DDAFusion_" + timestamp + "_" + train_phase + '.pth'))
