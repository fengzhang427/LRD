import torch
import rawpy
import numpy as np
import os
from tqdm import tqdm
import argparse
from raw_process.SID_post_process import load_CRF, raw2rgb_postprocess, tensor2im, quality_assess
from datasets.Denoise_dataset import pack_raw_SID, pack_raw_SID_dark_shading
from utils import IlluminanceCorrect, get_config
from models.denoise_model import UNetSeeInDark
import torchvision.utils as vutils


SonyCCM = np.array([[1.9712269, -0.6789218, -0.29230508],
                    [-0.29104823, 1.748401, -0.45735288],
                    [0.02051281, -0.5380369, 1.5175241]])


def denoise_test(denoise_model, data_list, data_dir, save_path, set_ratio, set_corrector):
    img_info = []
    with open(data_list, 'r') as f:
        for i, img_pair in enumerate(f):
            img_pair = img_pair.strip()  # ./Sony/short/10003_00_0.04s.ARW ./Sony/long/10003_00_10s.ARW ISO200 F9
            img_file, lbl_file, iso, focus = img_pair.split(' ')
            ISO = int(''.join([x for x in iso if x.isdigit()]))
            img_exposure = float(os.path.split(img_file)[-1][9:-5])  # 0.04
            lbl_exposure = float(os.path.split(lbl_file)[-1][9:-5])  # 10
            ratio = min(lbl_exposure / img_exposure, 300)
            if ratio == set_ratio:
                img_info.append({
                    'img': img_file,
                    'lbl': lbl_file,
                    'img_exposure': img_exposure,
                    'lbl_exposure': lbl_exposure,
                    'ratio': ratio,
                    'iso': ISO,
                    'focus': focus,
                })
    total_iter = 0
    psnr = 0
    ssim = 0
    count = 0

    for idx, info in tqdm(enumerate(img_info), total=len(img_info), desc='Train epoch={}, iter={}'.format(0, total_iter), ncols=80, leave=False):
        noisy_fn, clean_fn, iso, ratio = info['img'], info['lbl'], info['iso'], info['ratio']
        total_iter += 1
        count += 1

        clean_path = os.path.join(data_dir, clean_fn)
        noisy_path = os.path.join(data_dir, noisy_fn)

        raw_clean = rawpy.imread(clean_path)
        clean_image = pack_raw_SID(raw_clean)
        raw_noisy = rawpy.imread(noisy_path)
        # noisy_image = pack_raw_bayer(raw_noisy) * ratio
        noisy_image = pack_raw_SID_dark_shading(raw_noisy, iso) * ratio

        noisy_image = np.maximum(np.minimum(noisy_image, 1.0), 0)
        noisy_image = np.ascontiguousarray(noisy_image)
        clean_image = np.ascontiguousarray(clean_image)

        clean_tensor = torch.from_numpy(clean_image).unsqueeze(0).cuda()
        noisy_tensor = torch.from_numpy(noisy_image).unsqueeze(0).cuda()

        denoise_model.eval()
        with torch.no_grad():
            denoised_tensor = denoise_model(noisy_tensor)
            # denoised_tensor = set_corrector(denoised_tensor, clean_tensor)
            clean_numpy = tensor2im(clean_tensor)
            denoised_numpy = tensor2im(denoised_tensor)
            res = quality_assess(denoised_numpy, clean_numpy, data_range=255)
            psnr += res['PSNR']
            ssim += res['SSIM']

            CRF = load_CRF()
            with rawpy.imread(noisy_path) as raw:
                if not os.path.exists('%s/denoised/' % save_path):
                    print("Creating directory: {}".format('%s/denoised/' % save_path))
                    os.makedirs('%s/denoised/' % save_path)
                if not os.path.exists('%s/noisy/' % save_path):
                    print("Creating directory: {}".format('%s/noisy/' % save_path))
                    os.makedirs('%s/noisy/' % save_path)
                if not os.path.exists('%s/clean/' % save_path):
                    print("Creating directory: {}".format('%s/clean/' % save_path))
                    os.makedirs('%s/clean/' % save_path)
                clean_image = raw2rgb_postprocess(clean_tensor, raw, CRF)
                noisy_image = raw2rgb_postprocess(noisy_tensor, raw, CRF)
                denoised_image = raw2rgb_postprocess(denoised_tensor, raw, CRF)
                vutils.save_image(denoised_image.data, '%s/denoised/%s_%s.png' % (save_path, total_iter, ratio))
                vutils.save_image(noisy_image.data, '%s/noisy/%s_%s.png' % (save_path, total_iter, ratio))
                vutils.save_image(clean_image.data, '%s/clean/%s_%s.png' % (save_path, total_iter, ratio))
    print("===> PSNR: {}, SSIM:{}".format(psnr / count, ssim / count))
    print("Total test raw image number: %s" % total_iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoise args setting')
    parser.add_argument('--config', type=str, default='./configs/Noise_Synthesis_SID.yaml', help='Path to the config file.')
    parser.add_argument('--checkpoints_denoise', type=str, default='./checkpoints_denoise/SonyA7S2_Ours_best_model.pth', help='Path to the config file.')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--seed', type=int, default=3407, help="random seed")
    parser.add_argument('--ratio', type=int, default=300, help="exposure ratio")
    opts = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    config = get_config(opts.config)

    corrector = IlluminanceCorrect()
    print(opts.checkpoints_denoise)
    print(opts.ratio)

    raw_dir = config['SID_root']
    test_img_list = config['test_img_list']

    save_dir = './SID_denoise_results_ratio%s/' % opts.ratio
    if not os.path.exists(save_dir):
        print("Creating directory: {}".format(save_dir))
        os.makedirs(save_dir)

    denoise_model = UNetSeeInDark(in_channels=4, out_channels=4).cuda()
    state_dict_denoise = torch.load(opts.checkpoints_denoise, map_location='cpu')
    denoise_model.load_state_dict(state_dict_denoise)
    denoise_test(denoise_model, test_img_list, raw_dir, save_dir, opts.ratio, corrector)
