"""Forward processing of raw data to sRGB images.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import numpy as np
import torch
from torchinterp1d import Interp1d
from os.path import join
import rawpy
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def apply_gains(bayer_images, wbs):
    """Applies white balance to a batch of Bayer images."""
    N, C, _, _ = bayer_images.shape
    outs = bayer_images * wbs.view(1, C, 1, 1).repeat_interleave(N, dim=0)
    return outs


def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    images = images.permute(
        0, 2, 3, 1)  # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images[:, :, :, None, :]
    ccms = ccms[:, None, None, :, :]
    outs = torch.sum(images * ccms, dim=-1)
    # Re-Permute the tensor back to BxCxHxW format
    outs = outs.permute(0, 3, 1, 2)
    return outs


def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    outs = torch.clamp(images, min=1e-8) ** (1 / gamma)
    # outs = (1 + gamma[0]) * np.power(images, 1.0/gamma[1]) - gamma[0] + gamma[2]*images
    outs = torch.clamp((outs * 255).int(), min=0, max=255).float() / 255
    return outs


def binning(bayer_images):
    """RGBG -> RGB"""
    lin_rgb = torch.stack([
        bayer_images[:, 0, ...],
        torch.mean(bayer_images[:, [1, 3], ...], dim=1),
        bayer_images[:, 2, ...]], dim=1)
    return lin_rgb


def process(bayer_images, wbs, cam2rgbs, gamma=2.2, CRF=None):
    """Processes a batch of Bayer RGBG images into sRGB images."""
    # White balance.
    bayer_images = apply_gains(bayer_images, wbs)
    # Binning
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    images = binning(bayer_images)
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = torch.clamp(images, min=0.0, max=1.0)
    if CRF is None:
        images = gamma_compression(images, gamma)
    else:
        images = camera_response_function(images, CRF)

    return images


def camera_response_function(images, CRF):
    E, fs = CRF  # unpack CRF data

    outs = torch.zeros_like(images)
    device = images.device

    for i in range(images.shape[0]):
        img = images[i].view(3, -1)
        out = Interp1d()(E.to(device), fs.to(device), img)
        outs[i, ...] = out.view(3, images.shape[2], images.shape[3])

    outs = torch.clamp((outs * 255).int(), min=0, max=255).float() / 255
    return outs


def raw2rgb(packed_raw, raw, CRF=None, gamma=2.2):
    """Raw2RGB pipeline (preprocess version)"""
    wb = np.array(raw.camera_whitebalance)
    wb /= wb[1]
    cam2rgb = raw.rgb_camera_matrix[:3, :3]

    if isinstance(packed_raw, np.ndarray):
        packed_raw = torch.from_numpy(packed_raw).float()

    wb = torch.from_numpy(wb).float().to(packed_raw.device)
    cam2rgb = torch.from_numpy(cam2rgb).float().to(packed_raw.device)

    out = process(packed_raw[None], wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=gamma, CRF=CRF)[0, ...].numpy()

    return out


def raw2rgb_v2(packed_raw, wb, ccm, CRF=None, gamma=2.2):  # RGBG
    packed_raw = torch.from_numpy(packed_raw).float()
    wb = torch.from_numpy(wb).float()
    cam2rgb = torch.from_numpy(ccm).float()
    out = process(packed_raw[None], wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=gamma, CRF=CRF)[0, ...].numpy()
    return out


def raw2rgb_postprocess(packed_raw, raw, CRF=None):
    """Raw2RGB pipeline (postprocess version)"""
    assert packed_raw.ndimension() == 4
    wb = np.array(raw.camera_whitebalance)
    wb /= wb[1]
    cam2rgb = raw.rgb_camera_matrix[:3, :3]

    wb = torch.from_numpy(wb[None]).float().to(packed_raw.device)
    cam2rgb = torch.from_numpy(cam2rgb[None]).float().to(packed_raw.device)
    out = process(packed_raw, wbs=wb, cam2rgbs=cam2rgb, gamma=2.2, CRF=CRF)
    return out


def read_wb_ccm(raw):
    wb = np.array(raw.camera_whitebalance)
    wb /= wb[1]
    wb = wb.astype(np.float32)
    ccm = raw.rgb_camera_matrix[:3, :3].astype(np.float32)
    return wb, ccm


def read_emor(address):
    def _read_curve(lst):
        curve = [l.strip() for l in lst]
        curve = ' '.join(curve)
        curve = np.array(curve.split()).astype(np.float32)
        return curve

    with open(address) as f:
        lines = f.readlines()
        k = 1
        E = _read_curve(lines[k:k + 256])
        k += 257
        f0 = _read_curve(lines[k:k + 256])
        hs = []
        for _ in range(25):
            k += 257
            hs.append(_read_curve(lines[k:k + 256]))

        hs = np.array(hs)

        return E, f0, hs


def read_dorf(address):
    with open(address) as f:
        lines = f.readlines()
        curve_names = lines[0::6]
        Es = lines[3::6]
        Bs = lines[5::6]

        Es = [np.array(E.strip().split()).astype(np.float32) for E in Es]
        Bs = [np.array(B.strip().split()).astype(np.float32) for B in Bs]

    return curve_names, Es, Bs


def load_CRF():
    # init CRF function
    fs = np.loadtxt(join('raw_process/EMoR', 'CRF_SonyA7S2_5.txt'))
    E, _, _ = read_emor(join('raw_process/EMoR', 'emor.txt'))
    E = torch.from_numpy(E).repeat(3, 1)
    fs = torch.from_numpy(fs)
    CRF = (E, fs)
    return CRF


def tensor2im(image_tensor, visualize=False, video=False):
    image_tensor = image_tensor.detach()

    if visualize:
        image_tensor = image_tensor[:, 0:3, ...]

    if not video:
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    else:
        image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1))) * 255.0

    image_numpy = np.clip(image_numpy, 0, 255)

    return image_numpy


def quality_assess(X, Y, data_range=255):
    # Y: correct; X: estimate
    if X.ndim == 3:
        psnr = compare_psnr(Y, X, data_range=data_range)
        ssim = compare_ssim(Y, X, data_range=data_range, multichannel=True)
        return {'PSNR':psnr, 'SSIM': ssim}
    else:
        raise NotImplementedError


def postprocess_bayer(rawpath, img4c):
    img4c = img4c.detach()
    img4c = img4c[0].cpu().float().numpy()
    img4c = np.clip(img4c, 0, 1)

    # unpack 4 channels to Bayer image
    raw = rawpy.imread(rawpath)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern == 0)
    G1 = np.where(raw_pattern == 1)
    G2 = np.where(raw_pattern == 3)
    B = np.where(raw_pattern == 2)

    black_level = np.array(raw.black_level_per_channel)[:, None, None]

    white_point = 16383

    img4c = img4c * (white_point - black_level) + black_level

    img_shape = raw.raw_image_visible.shape
    H = img_shape[0]
    W = img_shape[1]

    raw.raw_image_visible[R[0][0]:H:2, R[1][0]:W:2] = img4c[0, :, :]
    raw.raw_image_visible[G1[0][0]:H:2, G1[1][0]:W:2] = img4c[1, :, :]
    raw.raw_image_visible[B[0][0]:H:2, B[1][0]:W:2] = img4c[2, :, :]
    raw.raw_image_visible[G2[0][0]:H:2, G2[1][0]:W:2] = img4c[3, :, :]

    # out = raw.postprocess(use_camera_wb=False, user_wb=[1,1,1,1], half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    # out = raw.postprocess(use_camera_wb=False, user_wb=[1.96875, 1, 1.444, 1], half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    out = raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=8, bright=1,
                          user_black=None, user_sat=None)
    return out
