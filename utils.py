import functools
import math
import pickle
import imageio
import glob
import random
import h5py
import rawpy
import scipy.io as sio
import time
import numpy as np
import skimage.metrics
import torch
import yaml
import torchvision.utils as vutils
from PIL import Image
import os
from torch import nn
from torch.nn import init, Identity
from torch.optim import lr_scheduler
from torch.autograd import Variable
from models.denoise_model import vgg_19
from skimage import io, color
import colour
import cv2
import torch.nn.functional as F
from pylab import histogram, interp


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_psnr(im1, im2):
    return skimage.metrics.peak_signal_noise_ratio(im1, im2, data_range=255)


def get_ssim(im1, im2):
    return skimage.metrics.structural_similarity(im1, im2, data_range=255, gaussian_weights=True,
                                                 use_sample_covariance=False, multichannel=True)


def write_loss(iterations, trainer, train_writer):
    member_loss = [attr for attr in dir(trainer)
                   if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr)]
    member_grad = [attr for attr in dir(trainer)
                   if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('grad' in attr)]
    member_nwd = {attr: getattr(trainer, attr) for attr in dir(trainer)
                  if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('nwd' in attr)}
    member_kld = {attr: getattr(trainer, attr) for attr in dir(trainer)
                  if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('kld' in attr)}
    for loss in member_loss:
        train_writer.add_scalar('/loss/%s' % loss, getattr(trainer, loss), iterations + 1)
    for grad in member_grad:
        train_writer.add_scalar('/grad/%s' % grad, getattr(trainer, grad), iterations + 1)

    train_writer.add_scalars('/Discriminator/', member_nwd, iterations + 1)
    train_writer.add_scalars('/KL_Distance/', member_kld, iterations + 1)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]  # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=False)
    vutils.save_image(image_grid, file_name, nrow=1)


def init_weights(net, init_type='normal'):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    :param iterations:
    :param hyperparameters:
    :param optimizer:
    """
    if hyperparameters['lr_policy'] == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + hyperparameters['epoch_count'] - hyperparameters['n_epochs']) / float(
                hyperparameters['n_epochs_decay'] + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=iterations)
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    elif hyperparameters['lr_policy'] == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif hyperparameters['lr_policy'] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparameters['n_epochs'], eta_min=0,
                                                   last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, num_groups=3, num_channels=64, affine=True,
                                       track_running_stats=True)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm, elementwise_affine=True, track_running_stats=True)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_config(config):
    with open(config, 'r') as stream:
        loader = yaml.load(stream, Loader=yaml.FullLoader)
        return loader


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def raw2LRGB(bayer_images):
    """RGBG -> linear RGB"""
    lin_rgb = torch.stack([
        bayer_images[:, 0, ...],
        torch.mean(bayer_images[:, [1, 3], ...], dim=1),
        bayer_images[:, 2, ...]], dim=1)

    return lin_rgb


def open_hdf5(filename):
    while True:
        try:
            hdf5_file = h5py.File(filename, 'r')
            return hdf5_file
        except OSError as e:
            print(e)
            print(filename, ' waiting')
            time.sleep(3)  # Wait a bit


def extract_metainfo(path='0151_METADATA_RAW_010.MAT'):
    meta = sio.loadmat(path)['metadata']
    mat_vals = meta[0][0]
    mat_keys = mat_vals.dtype.descr

    keys = []
    for item in mat_keys:
        keys.append(item[0])

    py_dict = {}
    for key in keys:
        py_dict[key] = mat_vals[key]

    device = py_dict['Model'][0].lower()
    bitDepth = py_dict['BitDepth'][0][0]
    if 'iphone' in device or bitDepth != 16:
        noise = py_dict['UnknownTags'][-2][0][-1][0][:2]
        iso = py_dict['DigitalCamera'][0, 0]['ISOSpeedRatings'][0][0]
        pattern = py_dict['SubIFDs'][0][0]['UnknownTags'][0][0][1][0][-1][0]
        time = py_dict['DigitalCamera'][0, 0]['ExposureTime'][0][0]

    else:
        noise = py_dict['UnknownTags'][-1][0][-1][0][:2]
        iso = py_dict['ISOSpeedRatings'][0][0]
        pattern = py_dict['UnknownTags'][1][0][-1][0]
        time = py_dict['ExposureTime'][0][0]  # the 0th row and 0th line item

    rgb = ['R', 'G', 'B']
    pattern = ''.join([rgb[i] for i in pattern])

    asShotNeutral = py_dict['AsShotNeutral'][0]
    b_gain, _, r_gain = asShotNeutral

    # only load ccm1
    # ccm = py_dict['ColorMatrix1'][0].astype(float).reshape((3, 3))
    ccm = py_dict['ColorMatrix2'][0].astype(float).reshape((3, 3))

    return {'device': device,
            'pattern': pattern,
            'iso': iso,
            'noise': noise,
            'time': time,
            'wb': np.array([r_gain, 1, b_gain]),
            'ccm': ccm, }


def transform_to_rggb(img, pattern):
    assert len(img.shape) == 2 and type(img) == np.ndarray

    if pattern.lower() == 'bggr':  # same pattern
        img = np.roll(np.roll(img, 1, axis=1), 1, axis=0)
    elif pattern.lower() == 'rggb':
        pass
    elif pattern.lower() == 'grbg':
        img = np.roll(img, 1, axis=1)
    elif pattern.lower() == 'gbrg':
        img = np.roll(img, 1, axis=0)
    else:
        assert 'no support'

    return img


def read_paired_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [tuple(fn.strip().split(' ')) for fn in fns]
    return fns


def stack2raw(var):
    _, h, w = var.shape
    if var.is_cuda:
        res = torch.cuda.FloatTensor(h * 2, w * 2)
    else:
        res = torch.FloatTensor(h * 2, w * 2)
    res[0::2, 0::2] = var[0]
    res[0::2, 1::2] = var[1]
    res[1::2, 0::2] = var[2]
    res[1::2, 1::2] = var[3]
    return res


def raw2stack(var):
    h, w = var.shape
    if var.is_cuda:
        res = torch.cuda.FloatTensor(4, h // 2, w // 2).fill_(0)
    else:
        res = torch.FloatTensor(4, h // 2, w // 2).fill_(0)
    res[0] = var[0::2, 0::2]
    res[1] = var[0::2, 1::2]
    res[2] = var[1::2, 0::2]
    res[3] = var[1::2, 1::2]
    return res


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:  # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:  # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)  # collect all the images and return
        return return_images


def create_gif(im_dir):
    root = '/home/abdo/skynet/_Code/fourier_flows/experiments/sidd/UNCOND/'
    sample_id = 50
    im_dir = os.path.join(root, 'one_sample_%04d' % sample_id)
    filenames = glob.glob(os.path.join(im_dir, '*.png'))
    filenames = sorted(filenames)
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(im_dir, 'sample_%04d.gif' % sample_id), images, duration=1)


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


def get_histogram(data, bin_edges=None, left_edge=0.0, right_edge=1.0, n_bins=1000):
    data_range = right_edge - left_edge
    bin_width = data_range / n_bins
    if bin_edges is None:
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + (bin_width / 2.0)
    n = np.prod(data.shape)
    hist, _ = np.histogram(data, bin_edges)
    return hist / n, bin_centers


def cal_kld(p_data, q_data, left_edge=0.0, right_edge=1.0, n_bins=1000):
    """Returns forward, inverse, and symmetric KL divergence between two sets of data points p and q"""
    bw = 0.2 / 64
    bin_edges = np.concatenate(([-1000.0], np.arange(-0.1, 0.1 + 1e-9, bw), [1000.0]), axis=0)
    # bin_edges = None
    p, _ = get_histogram(p_data, bin_edges, left_edge, right_edge, n_bins)
    q, _ = get_histogram(q_data, bin_edges, left_edge, right_edge, n_bins)
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    logp = np.log(p)
    logq = np.log(q)
    kl_fwd = np.sum(p * (logp - logq))
    kl_inv = np.sum(q * (logq - logp))
    kl_sym = (kl_fwd + kl_inv) / 2.0
    return kl_sym  # kl_fwd #, kl_inv, kl_sym


def kl_div_3_data(p_data, q_data, bin_edges=None, left_edge=0.0, right_edge=1.0, n_bins=1000):
    """Returns forward, inverse, and symmetric KL divergence between two sets of data points p and q"""
    if bin_edges is None:
        data_range = right_edge - left_edge
        bin_width = data_range / n_bins
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    p, _ = get_histogram(p_data, bin_edges, left_edge, right_edge, n_bins)
    q, _ = get_histogram(q_data, bin_edges, left_edge, right_edge, n_bins)
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    logp = np.log(p)
    logq = np.log(q)
    kl_fwd = np.sum(p * (logp - logq))
    kl_inv = np.sum(q * (logq - logp))
    kl_sym = (kl_fwd + kl_inv) / 2.0
    return kl_fwd, kl_inv, kl_sym


def compute_gradient_penalty2d(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)), dtype=real_samples.dtype,
                         device=real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)[..., 0]
    fake = Variable(torch.cuda.FloatTensor(d_interpolates.shape[0], 1).fill_(1.0), requires_grad=False).view(-1)

    # print(d_interpolates.shape, interpolates.shape, fake.shape)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
        interpolatesv = real_data
    elif type == 'fake':
        interpolatesv = fake_data
    elif type == 'mixed':
        alpha = torch.rand(real_data.shape[0], 1, device=device)
        alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
            *real_data.shape)
        interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
    else:
        raise NotImplementedError('{} not implemented'.format(type))
    interpolatesv.requires_grad_(True)
    disc_interpolates = netD(interpolatesv)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)
    gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
    gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean()  # added eps
    return gradient_penalty


class IlluminanceCorrect(nn.Module):
    def __init__(self):
        super(IlluminanceCorrect, self).__init__()

    # Illuminance Correction
    def forward(self, predict, source):
        if predict.shape[0] != 1:
            output = torch.zeros_like(predict)
            if source.shape[0] != 1:
                for i in range(predict.shape[0]):
                    output[i:i + 1, ...] = self.correct(predict[i:i + 1, ...], source[i:i + 1, ...])
            else:
                for i in range(predict.shape[0]):
                    output[i:i + 1, ...] = self.correct(predict[i:i + 1, ...], source)
        else:
            output = self.correct(predict, source)
        return output

    def correct(self, predict, source):
        N, C, H, W = predict.shape
        predict = torch.clamp(predict, 0, 1)
        assert N == 1
        output = torch.zeros_like(predict, device=predict.device)
        pred_c = predict[source != 1]
        source_c = source[source != 1]

        num = torch.dot(pred_c, source_c)
        den = torch.dot(pred_c, pred_c)
        output = num / den * predict
        # print(num / den)

        return output


def kl_gauss_zero_center(sigma_fake, sigma_real):
    '''
    Input:
        sigma_fake: 1 x C x H x W, torch array
        sigma_real: 1 x C x H x W, torch array
    '''
    div_sigma = torch.div(sigma_fake, sigma_real)
    div_sigma.clamp_(min=0.1, max=10)
    log_sigma = torch.log(1 / div_sigma)
    distance = 0.5 * torch.mean(log_sigma + div_sigma - 1.)
    return distance


def estimate_sigma_gauss(img_noisy, img_gt):
    win_size = 7
    err2 = (img_noisy - img_gt) ** 2
    kernel = get_gausskernel(win_size, chn=4).to(img_gt.device)
    sigma = gaussblur(err2, kernel, win_size, chn=4)
    sigma.clamp_(min=1e-10)

    return sigma


def get_gausskernel(p, chn=4):
    '''
    Build a 2-dimensional Gaussian filter with size p
    '''
    x = cv2.getGaussianKernel(p, sigma=-1)   # p x 1
    y = np.matmul(x, x.T)[np.newaxis, np.newaxis,]  # 1x 1 x p x p
    out = np.tile(y, (chn, 1, 1, 1)) # chn x 1 x p x p

    return torch.from_numpy(out).type(torch.float32)


def gaussblur(x, kernel, p=5, chn=4):
    x_pad = F.pad(x, pad=[int((p-1)/2),]*4, mode='reflect')
    y = F.conv2d(x_pad, kernel, padding=0, stride=1, groups=chn)

    return y
