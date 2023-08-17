import gc
import time
import torch
import os
import sys
import torch.backends.cudnn as cudnn
import argparse
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets.Denoise_dataset import NPYDataset_LRD
from models.denoise_model import UNetSeeInDark
import torch.nn.functional as F
from utils import get_config
from raw_process.SID_post_process import tensor2im, quality_assess
import numpy as np

# parse options
parser = argparse.ArgumentParser(description='Low-Light Raw Noise Synthesis args setting')
parser.add_argument('--config', type=str, default='./configs/Noise_Synthesis_LRD.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='./Denoise_results_LRD', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--seed', type=int, default=3407, help="random seed")
parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--patch_size', type=int, default=512, help='crop patch size')
parser.add_argument('--log_ids', type=str, default='exp0', help='log ID')
parser.add_argument('--height', type=int, default=512, help='input height size')
parser.add_argument('--width', type=int, default=512, help='input width size')
parser.add_argument('--learning_rate', type=int, default=0.0002, help='input width size')
parser.add_argument('--synthesis_root', default='/home/dancer/train_datasets/Generated_train_samples_LRD/')
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--image_save_epoch', type=int, default=10)
parser.add_argument('--snapshot_save_epoch', type=int, default=10)

opts = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
config = get_config(opts.config)


def stack2raw_RGGB(var):
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

def raw2stack_RGGB(var):
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


def forward_patches(model, input_img, patch_size=256, pad=32):
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


def main():
    cudnn.benchmark = True
    # Loss functions
    device = torch.device("cuda" if len(opts.gpu_ids) > 0 else "cpu")

    criterion_pixel = torch.nn.L1Loss().to(device)

    # model set and data loader
    model = UNetSeeInDark(in_channels=4, out_channels=4).to(device)

    # set logger and output folder
    writer = SummaryWriter(os.path.join(opts.output_path + "/logs/" + opts.log_ids))
    output_directory = os.path.join(opts.output_path + "/outputs")

    # set checkpoint folder
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)

    # set image folder
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)

    train_set = NPYDataset_LRD(opts.synthesis_root, config['test_img_list'], mode='train', patch_size=opts.patch_size, crop=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=opts.batch_size, shuffle=True, drop_last=True,
                                               num_workers=0, pin_memory=True)
    val_set = NPYDataset_LRD(config['LRD_root'], config['test_img_list'], mode='test', patch_size=opts.patch_size, crop=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    epoch_iter = 0
    total_iters = 0  # the total number of training iterations
    f = open("%s/Metrics_results.txt" % output_directory, 'a')
    f.truncate(0)

    print('start training')
    print(opts.synthesis_root)

    # init_weights(model, init_type='kaiming')
    optim = torch.optim.Adam(model.parameters(), lr=opts.learning_rate, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[15000, 25000], gamma=0.5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100, 180], gamma=0.5)
    model.train()

    for epoch in trange(epoch_iter, opts.max_epoch, desc='Train', ncols=80):
        epoch_start_time = time.time()  # timer for entire epoch
        for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader),
                                    desc='Train epoch={}, iter={}'.format(epoch_iter, total_iters), ncols=80,
                                    leave=False):
            total_iters += 1
            gc.collect()
            clean_raw = data['clean'].to(device)
            noise_raw = data['noisy'].to(device)
            optim.zero_grad()

            denoise_raw = model(noise_raw)

            loss = criterion_pixel(denoise_raw, clean_raw)
            loss.backward()
            optim.step()
            writer.add_scalar('Training/loss', loss, epoch + 1)
        scheduler.step()

        if (epoch + 1) % opts.image_save_epoch == 0:
            print("[*] Evaluating for phase train / epoch %d..." % (epoch + 1))
            psnr = 0
            ssim = 0
            count = 0
            for batch_idx, data in tqdm(enumerate(val_loader), total=len(val_loader), desc='Valid', ncols=80,
                                        leave=False):
                gc.collect()
                clean_raw = data['clean'].to(device)
                noise_raw = data['noisy'].to(device)
                clean_path = data['clean_path'][0]
                model.eval()
                with torch.no_grad():
                    denoise_raw = forward_patches(model, noise_raw)
                    clean_numpy = tensor2im(clean_raw)
                    denoised_numpy = tensor2im(denoise_raw)
                    res = quality_assess(denoised_numpy, clean_numpy, data_range=255)
                    psnr += res['PSNR']
                    ssim += res['SSIM']
                    count += 1
            metrics_data = ["Epoch: %s, PSNR: %s, SSIM: %s" % (epoch + 1, psnr / count, ssim / count), '\n']
            f.writelines(metrics_data)
            print("===> Iteration[{}]: psnr: {}, ssim:{}".format(epoch + 1, psnr / count, ssim / count))

        if (epoch + 1) % opts.snapshot_save_epoch == 0:
            torch.save(model.state_dict(), checkpoint_directory + '/check_' + str(epoch + 1) + '.pth')

        if epoch + 1 >= opts.max_epoch:
            writer.close()
            end_time = time.time() - epoch_start_time
            print("Finish training in [%4.4f]" % end_time)
            sys.exit('Finish training')


if __name__ == '__main__':
    main()
