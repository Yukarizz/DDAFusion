import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia import color
from .contextual import contextual
from kornia import color

degradations = ['foggy lightly','foggy moderately','foggy heavily','rainy lightly',
                'rainy moderately','rainy heavily','snowy lightly','snowy moderately','snowy heavily',
                'raindrop lightly','raindrop moderately','raindrop heavily','common']
def angle(a, b):
    return (1-F.cosine_similarity(a, b, dim=1)).unsqueeze(1).mean()
class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()
        # self.Context_loss = contextual.ContextualLoss().cuda()
        self.balance = [1,10,0.1]

    def forward(self,image_vis,image_ir,generate_img,corse_fusion):
        # contextual_loss = self.Context_loss(generate_img, image_vis)
        ycbcr_vi = color.rgb_to_ycbcr(image_vis)
        ycbcr_ir = color.rgb_to_ycbcr(image_ir)
        ycbcr_fu = color.rgb_to_ycbcr(generate_img)
        image_y_vi = ycbcr_vi[:,:1,:,:]
        image_cbcr_vi = ycbcr_vi[:,1:,:,:]
        image_y_ir = ycbcr_ir[:, :1, :, :]
        image_y_fu = ycbcr_fu[:, :1, :, :]
        image_cbcr_fu = ycbcr_fu[:, 1:, :, :]
        x_in_max = torch.max(image_y_vi,image_y_ir)

        highlight_mask = (image_y_ir>=image_y_vi)*1.

        loss_in = F.l1_loss(highlight_mask*image_y_ir,image_y_fu) + F.l1_loss(((1-highlight_mask)*image_vis),generate_img)
        y_grad = self.sobelconv(image_y_vi)
        ir_grad = self.sobelconv(image_y_ir)
        generate_img_grad = self.sobelconv(image_y_fu)
        x_grad_joint = torch.max(y_grad,ir_grad)
        loss_grad = F.l1_loss(x_grad_joint,generate_img_grad)
        # image_y_corse = color.rgb_to_ycbcr(corse_fusion)[:, :1, :, :]
        # loss_ssim = corr_loss(image_y_ir, image_y_vi, image_y_corse)
        color_anlge_loss = 5*F.l1_loss(image_cbcr_fu,image_cbcr_vi)
        loss_total = self.balance[0]*loss_in + self.balance[1]*loss_grad + color_anlge_loss
        return loss_total,self.balance[0]*loss_in,self.balance[1]*loss_grad, color_anlge_loss

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()

def corr_loss(image_ir, img_vis, img_fusion, eps=1e-6):
    reg = REG()
    corr = reg(image_ir, img_vis, img_fusion)
    corr_loss = 1./(corr + eps)
    return corr_loss

class REG(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """
    def __init__(self):
        super(REG, self).__init__()

    def corr2(self, img1, img2):
        img1 = img1 - img1.mean()
        img2 = img2 - img2.mean()
        r = torch.sum(img1*img2)/torch.sqrt(torch.sum(img1*img1)*torch.sum(img2*img2))
        return r

    def forward(self, a, b, c):
        return self.corr2(a, c) + self.corr2(b, c)

def clip_degra_loss(text_features, fimage_degra_context):
    BCE = torch.nn.BCELoss()
    text_features = text_features.cuda()
    B, D = fimage_degra_context.shape
    label = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]).float().cuda().repeat_interleave(B, dim=0)
    fimage_degra_context /= fimage_degra_context.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    corr_matrix = (100 * fimage_degra_context @ text_features.T).softmax(-1)
    loss = BCE(corr_matrix,label)
    index = torch.argmax(corr_matrix[0])
    return loss, degradations[index] +" confidence " + str(round(corr_matrix[0][index].item(),2))