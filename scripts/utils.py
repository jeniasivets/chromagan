import torch, torch.nn as nn, torch.nn.functional as F
from skimage.color import lab2rgb, rgb2lab

def kl_divergence(y_true, y_pred):
    return torch.mean(torch.sum(torch.exp(y_true) * (y_true - y_pred), dim=1)) 

def compute_gradient_penalty(D, real_samples, fake_samples):
    device = next(D.parameters()).device
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.shape, requires_grad=False, device=device), # gradients w.t. output. 1 is default
        create_graph=True,
        retain_graph=True, # keep all gradients for further optimization steps
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def rgb_to_lab(img):
    lab_img = rgb2lab(img.permute(0, 3, 2, 1))
    lab_img[..., 0] = lab_img[..., 0] / 100
    lab_img[..., 1] = (lab_img[..., 1] + 87) / 186
    lab_img[..., 2] = (lab_img[..., 2] + 108) / 203
    lab_img = torch.from_numpy(lab_img).permute(0, 3, 2, 1)
    return lab_img


def lab_to_rgb(img):
    b,c,h,w = img.shape
    rgb_img = img.detach().cpu().permute(0, 2, 3, 1).flatten(end_dim=1)
    rgb_img[..., 0] = rgb_img[..., 0] * 100
    rgb_img[..., 1] = (rgb_img[..., 1]  * 186) - 87
    rgb_img[..., 2] = (rgb_img[..., 2] * 203) - 108
    return torch.tensor(lab2rgb(rgb_img), dtype=torch.float).reshape((b,h,w,c)).permute(0, 3, 1, 2)


def compute_psnr(img_pred, img_gt):
    img_pred = img_pred.detach().cpu()
    img_gt = img_gt.detach().cpu()
    b, c, h, w = img_gt.shape
    mse = 1 / (c * h * w) * torch.sum((img_pred - img_gt)**2, dim=(1,2,3))
    psnr = 10 * torch.log10((img_gt**2).max(-1)[0].max(-1)[0].max(-1)[0] / mse).mean().item()
    return psnr


