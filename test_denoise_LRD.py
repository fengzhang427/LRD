import torch
import rawpy
import numpy as np
import os
from tqdm import tqdm
import argparse
from torch.nn import functional as F
from raw_process.SID_post_process import tensor2im, quality_assess
import torchvision.utils as vutils
import glob
from raw_process import LRD_post_process as process
from models.denoise_model import UNetSeeInDark
from utils import get_config
from datasets.Denoise_dataset import pack_raw_LRD, metainfo


def forward_patches_denoise(model, input_img, patch_size=256, pad=32):
    shift = patch_size - pad * 2

    input_img = F.pad(input_img, (pad, pad, pad, pad), mode='reflect')
    denoised = torch.zeros_like(input_img)

    _, _, H, W = input_img.shape
    for i in np.arange(0, H, shift):
        for j in np.arange(0, W, shift):
            h_end, w_end = min(i + patch_size, H), min(j + patch_size, W)
            h_start, w_start = h_end - patch_size, w_end - patch_size

            input_var = input_img[..., h_start: h_end, w_start: w_end]
            with torch.no_grad():
                out_var = model(input_var)
            denoised[..., h_start + pad: h_end - pad, w_start + pad: w_end - pad] = \
                out_var[..., pad:-pad, pad:-pad]

    denoised = denoised[..., pad:-pad, pad:-pad]
    denoised = denoised.clip(0, 1)
    return denoised


def denoise_test_unpair(denoise_model, data_dir):
    img_list_noisy = sorted(glob.glob(os.path.join(data_dir, 'test_set', '*.DNG')))
    total_iter = 0

    for idx, noisy_image in tqdm(enumerate(img_list_noisy), total=len(img_list_noisy), desc='Train epoch={}, iter={}'.format(0, total_iter), ncols=80, leave=False):
        total_iter += 1

        raw_noisy = rawpy.imread(img_list_noisy[idx])
        noisy_image = pack_raw_LRD(raw_noisy)

        ISO, EXPO, EV, cshot, cread = metainfo(img_list_noisy[idx])

        ratio_EV = 1 / 2 ** EV
        noisy_image = noisy_image * ratio_EV

        noisy_image = np.maximum(np.minimum(noisy_image, 1.0), 0)
        noisy_image = np.ascontiguousarray(noisy_image)
        noisy_tensor = torch.from_numpy(noisy_image).unsqueeze(0).cuda()

        denoise_model.eval()
        with torch.no_grad():
            denoise_tensor = forward_patches_denoise(denoise_model, noisy_tensor)

            with rawpy.imread(img_list_noisy[idx]) as noisy_raw:
                camera_whitebalance = torch.Tensor(noisy_raw.camera_whitebalance)
                red_gain = camera_whitebalance[0].cuda()
                blue_gain = camera_whitebalance[2].cuda()
                ccm = [[1.359, -0.357, -0.001], [-0.209, 1.258, -0.049], [-0.049, -0.499, 1.548]]
                cam2rgb = torch.FloatTensor(ccm).unsqueeze(0).cuda()

                denoise_rbg_image = process.process(denoise_tensor, red_gains=red_gain, blue_gains=blue_gain, cam2rgbs=cam2rgb)
                file_name_denoise = './test_unpair_denoise_dir/noisy_denoised/%s.bmp' % total_iter
                vutils.save_image(denoise_rbg_image.data, file_name_denoise, nrow=1)
                noisy_rbg_image = process.process(noisy_tensor, red_gains=red_gain, blue_gains=blue_gain, cam2rgbs=cam2rgb)
                file_name_noisy = './test_unpair_denoise_dir/noisy_input/%s.bmp' % total_iter
                vutils.save_image(noisy_rbg_image.data, file_name_noisy, nrow=1)
    print("Total test raw image number: %s" % total_iter)


def denoise_test_pair(denoise_model, data_list, data_dir, save_path, set_ev):
    img_info = []
    with open(data_list, 'r') as f:
        for i, img_pair in enumerate(f):
            img_pair = img_pair.strip()  # ./Sony/short/10003_00_0.04s.ARW ./Sony/long/10003_00_10s.ARW ISO200 F9
            clean_file, noisy_file, iso, ev, _ = img_pair.split(' ')
            ISO = int(''.join([x for x in iso if x.isdigit()]))
            EV = -float(''.join([x for x in ev if x.isdigit()]))
            clean_exposure = float(os.path.split(clean_file)[-1][9:-5])  # 0.04
            noisy_exposure = float(os.path.split(noisy_file)[-1][9:-5])  # 10
            ratio = (100 * clean_exposure) / (ISO * noisy_exposure)
            if set_ev - 0.5 < EV < set_ev + 0.5:
                img_info.append({
                    'clean': clean_file,
                    'noisy': noisy_file,
                    'clean_exposure': clean_exposure,
                    'noisy_exposure': noisy_exposure,
                    'ratio': ratio,
                    'iso': ISO,
                    'ev': EV,
                })
    total_iter = 0
    psnr = 0
    ssim = 0
    count = 0
    KLD = []

    for idx, info in tqdm(enumerate(img_info), total=len(img_info),
                          desc='Train epoch={}, iter={}'.format(0, total_iter), ncols=80, leave=False):
        clean_fn, noisy_fn, iso, ratio, ev = info['clean'], info['noisy'], info['iso'], info['ratio'], info['ev']
        total_iter += 1
        count += 1

        clean_path = os.path.join(data_dir, clean_fn)
        noisy_path = os.path.join(data_dir, noisy_fn)

        clean_raw = rawpy.imread(clean_path)
        clean_image = pack_raw_LRD(clean_raw)
        noise_raw = rawpy.imread(noisy_path)
        noisy_image = pack_raw_LRD(noise_raw) * ratio

        noisy_image = np.maximum(np.minimum(noisy_image, 1.0), 0)
        noisy_image = np.ascontiguousarray(noisy_image)
        clean_image = np.ascontiguousarray(clean_image)

        clean_tensor = torch.from_numpy(clean_image).unsqueeze(0).cuda()
        noisy_tensor = torch.from_numpy(noisy_image).unsqueeze(0).cuda()

        denoise_model.eval()
        with torch.no_grad():
            denoised_tensor = forward_patches_denoise(denoise_model, noisy_tensor)
            clean_numpy = tensor2im(clean_tensor)
            denoised_numpy = tensor2im(denoised_tensor)
            res = quality_assess(denoised_numpy, clean_numpy, data_range=255)
            psnr += res['PSNR']
            ssim += res['SSIM']

            with rawpy.imread(clean_path) as clean_raw:
                camera_whitebalance = torch.Tensor(clean_raw.camera_whitebalance)
                red_gain = camera_whitebalance[0].cuda()
                blue_gain = camera_whitebalance[2].cuda()
                ccm = [[1.359, -0.357, -0.001], [-0.209, 1.258, -0.049], [-0.049, -0.499, 1.548]]
                cam2rgb = torch.FloatTensor(ccm).unsqueeze(0).cuda()

                if not os.path.exists('%s/denoised/' % save_path):
                    print("Creating directory: {}".format('%s/denoised/' % save_path))
                    os.makedirs('%s/denoised/' % save_path)
                if not os.path.exists('%s/noisy/' % save_path):
                    print("Creating directory: {}".format('%s/noisy/' % save_path))
                    os.makedirs('%s/noisy/' % save_path)
                if not os.path.exists('%s/clean/' % save_path):
                    print("Creating directory: {}".format('%s/clean/' % save_path))
                    os.makedirs('%s/clean/' % save_path)

                denoise_rbg_image = process.process(denoised_tensor, red_gains=red_gain, blue_gains=blue_gain, cam2rgbs=cam2rgb)
                vutils.save_image(denoise_rbg_image.data, '%s/denoised/%s_ISO%s_%sEV.bmp' % (save_path, total_iter, iso, ev), nrow=1)
                noisy_rbg_image = process.process(noisy_tensor, red_gains=red_gain, blue_gains=blue_gain, cam2rgbs=cam2rgb)
                vutils.save_image(noisy_rbg_image.data, '%s/noisy/%s_ISO%s_%sEV.bmp' % (save_path, total_iter, iso, ev), nrow=1)
                clean_rbg_image = process.process(clean_tensor, red_gains=red_gain, blue_gains=blue_gain, cam2rgbs=cam2rgb)
                vutils.save_image(clean_rbg_image.data, '%s/clean/%s_ISO%s_%sEV.bmp' % (save_path, total_iter, iso, ev), nrow=1)

    print("===> PSNR: {}, SSIM:{}".format(psnr / count, ssim / count))
    print("KL Distance: %s" % (np.sum(KLD) / total_iter))
    print("Total test raw image number: %s" % total_iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoise args setting')
    parser.add_argument('--config', type=str, default='./configs/Noise_Synthesis_LRD.yaml', help='Path to the config file.')
    parser.add_argument('--checkpoints_denoise', type=str, default='./checkpoints_denoise/IMX586_Ours_best_model.pth', help='Path to the config file.')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--seed', type=int, default=3407, help="random seed")
    parser.add_argument('--EV', type=int, default=-3, help="set up the target exposure bias")
    opts = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    config = get_config(opts.config)

    raw_dir = config['LRD_root']
    test_img_list = config['test_img_list']

    unpair_data_dir = '/data/data_ssd/DJI/'

    save_dir = './LRD_denoise_results_%sEV/' % opts.EV
    if not os.path.exists(save_dir):
        print("Creating directory: {}".format(save_dir))
        os.makedirs(save_dir)

    denoise_model = UNetSeeInDark(in_channels=4, out_channels=4).cuda()
    state_dict_denoise = torch.load(opts.checkpoints_denoise, map_location='cpu')
    denoise_model.load_state_dict(state_dict_denoise)
    denoise_test_pair(denoise_model, test_img_list, raw_dir, save_dir, opts.EV)
    # denoise_test_unpair(denoise_model, unpair_data_dir)
