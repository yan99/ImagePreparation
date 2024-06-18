# Sampling for diffusion results
import torch
import numpy as np
from DDVM_utils import forwardDiffusion
from DDVM_data import img_w, img_h

@torch.no_grad()
def sample_timestep(model, x_t, rgb, fd, t, device):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = forwardDiffusion.get_index_from_list(fd, fd.betas, t, x_t.shape)
    sqrt_one_minus_alpha_bar_t = forwardDiffusion.get_index_from_list(fd, fd.sqrt_one_minus_alpha_bar, t, x_t.shape)
    sqrt_one_alphas_t = forwardDiffusion.get_index_from_list(fd, fd.sqrt_one_alphas, t, x_t.shape)

    # Call model (current image - noise prediction)
    # Without N
    # N C H W
    input = torch.cat((rgb, x_t), 1).to(device)
    t = t.to(device)
    epsilon = model(input, t).cpu().detach()
    t = t.cpu().detach()
    model_mean = sqrt_one_alphas_t * (x_t - betas_t * epsilon / sqrt_one_minus_alpha_bar_t)
    var_t = forwardDiffusion.get_index_from_list(fd, fd.var, t, x_t.shape)

    if t == 0:
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(var_t) * noise

@torch.no_grad()
def sample_image(num, fd, model, rgb, device):
    # Sample noise
    # N C H W
    noisy_depth = torch.randn((1, 1, img_h, img_w), device = "cpu")
    num_images = num
    stepsize = int(fd.T/num_images)
    out_imgs = torch.zeros((num_images, 1, img_h, img_w), device = "cpu")

    for i in range(0,fd.T)[::-1]:
        if (i == fd.T - 1):
            noisy_depth = noisy_depth
        else:
            noisy_depth = denoised_depth

        t = torch.full((1,), i, device = "cpu", dtype=torch.long)
        denoised_depth = sample_timestep(model, noisy_depth, rgb, fd, t, device)
        # Edit: This is to maintain the natural range of the distribution
        denoised_depth = torch.clamp(denoised_depth, -1.0, 1.0)
        if i % stepsize == 0:
            out_imgs[int(i/stepsize)] = denoised_depth
    
    return out_imgs