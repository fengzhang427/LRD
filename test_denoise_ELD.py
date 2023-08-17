import torch
import numpy as np
import os
import rawpy
from tqdm import tqdm
import argparse
from utils import IlluminanceCorrect, get_config
from datasets.Denoise_dataset import ELDEvalDataset
from models.denoise_model import UNetSeeInDark
from raw_process.SID_post_process import load_CRF, raw2rgb_postprocess, tensor2im, quality_assess
import torchvision.utils as vutils


SonyCCM = np.array([[1.9712269, -0.6789218, -0.29230508],
                    [-0.29104823, 1.748401, -0.45735288],
                    [0.02051281, -0.5380369, 1.5175241]])


def denoise_test(denoise_model, data_dir, save_path, set_ratio, set_corrector):
    scenes = list(range(1, 10 + 1))
    img_ids_set = [[4, 9, 14], [5, 10, 15]]
    cameras = ['SonyA7S2']
    suffixes = ['.ARW']
    total_iter = 0
    psnr = 0
    ssim = 0

    for i, img_ids in enumerate(img_ids_set):
        print(img_ids)
        eval_datasets = [ELDEvalDataset(data_dir, camera_suffix, scenes=scenes, img_ids=img_ids) for camera_suffix in
                         zip(cameras, suffixes)]
        eval_dataloaders = torch.utils.data.DataLoader(eval_datasets[0], batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        for idx, info in tqdm(enumerate(eval_dataloaders), total=len(eval_dataloaders),
                              desc='Train epoch={}, iter={}'.format(0, total_iter), ncols=80, leave=False):
            noisy_tensor, clean_tensor, clean_path, noisy_path, iso, ratio = info['input'], info['target'], info['rawpath'][0], info['fn'][0], info['iso'][0], int(info['ratio'][0])
            noisy_tensor = noisy_tensor.cuda()
            clean_tensor = clean_tensor.cuda()

            if ratio == set_ratio:
                total_iter += 1
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
    print("===> PSNR: {}, SSIM:{}".format(psnr / total_iter, ssim / total_iter))
    print("Total test raw image number: %s" % total_iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoise args setting')
    parser.add_argument('--config', type=str, default='./configs/Noise_Synthesis_SID.yaml', help='Path to the config file.')
    parser.add_argument('--checkpoints_denoise', type=str, default='./checkpoints_denoise/SonyA7S2_Ours_best_model.pth', help='Path to the config file.')
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--seed', type=int, default=3407, help="random seed")
    parser.add_argument('--ratio', type=int, default=100, help="set up the exposure ratio")
    opts = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    config = get_config(opts.config)

    corrector = IlluminanceCorrect()
    print(opts.checkpoints_denoise)
    print(opts.ratio)

    raw_dir = config['ELD_root']
    save_dir = './ELD_denoise_results_ratio%s/' % opts.ratio
    if not os.path.exists(save_dir):
        print("Creating directory: {}".format(save_dir))
        os.makedirs(save_dir)

    denoise_model = UNetSeeInDark(in_channels=4, out_channels=4).cuda()
    state_dict_denoise = torch.load(opts.checkpoints_denoise, map_location='cpu')
    denoise_model.load_state_dict(state_dict_denoise)
    # model.load_state_dict(state_dict['netG'])
    denoise_test(denoise_model, raw_dir, save_dir, opts.ratio, corrector)
