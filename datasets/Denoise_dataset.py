import os
import random
import rawpy
import numpy as np
from os.path import join
import exifread
import torch.utils.data as data
import torch
import torchvision.transforms.functional as tf
import glob
import pickle


def metainfo(rawpath):
    with open(rawpath, 'rb') as f:
        tags = exifread.process_file(f)
        _, suffix = os.path.splitext(os.path.basename(rawpath))

        noise_profile = eval(str(tags['Image Tag 0xC761']))
        cshot, cread = float(noise_profile[0][0]), float(noise_profile[1][0])

        if suffix == '.dng':
            expo = eval(str(tags['Image ExposureTime']))
            iso = eval(str(tags['Image ISOSpeedRatings']))
            ev = eval(str(tags['Image ExposureBiasValue']))
        else:
            expo = eval(str(tags['EXIF ExposureTime']))
            iso = eval(str(tags['EXIF ISOSpeedRatings']))
            ev = eval(str(tags['EXIF ExposureBiasValue']))

        # print('ISO: {}, ExposureTime: {}'.format(iso, expo))
    return iso, expo, ev, cread, cshot


def metainfo_ELD(rawpath):
    with open(rawpath, 'rb') as f:
        tags = exifread.process_file(f)
        _, suffix = os.path.splitext(os.path.basename(rawpath))

        if suffix == '.dng':
            expo = eval(str(tags['Image ExposureTime']))
            iso = eval(str(tags['Image ISOSpeedRatings']))
        else:
            expo = eval(str(tags['EXIF ExposureTime']))
            iso = eval(str(tags['EXIF ISOSpeedRatings']))

        # print('ISO: {}, ExposureTime: {}'.format(iso, expo))
    return iso, expo


def pack_raw_SID(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern == 0)
    G1 = np.where(raw_pattern == 1)
    B = np.where(raw_pattern == 2)
    G2 = np.where(raw_pattern == 3)

    white_point = raw.white_level
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[R[0][0]:H:2, R[1][0]:W:2],  # RGBG
                    im[G1[0][0]:H:2, G1[1][0]:W:2],
                    im[B[0][0]:H:2, B[1][0]:W:2],
                    im[G2[0][0]:H:2, G2[1][0]:W:2]), axis=0).astype(np.float32)

    black_level = np.array(raw.black_level_per_channel)[:, None, None].astype(np.float32)

    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)

    return out


def pack_raw_LRD(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)

    white_point = 65535.0

    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[0:H:2, 0:W:2],  # RGGB
                    im[0:H:2, 1:W:2],
                    im[1:H:2, 0:W:2],
                    im[1:H:2, 1:W:2]), axis=0).astype(np.float32)
    black_level = np.array(raw.black_level_per_channel)[:, None, None].astype(np.float32)
    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)
    return out



def get_darkshading(iso):
    darkshading = {}
    with open(os.path.join('./resources', 'darkshading_BLE.pkl'), 'rb') as f:
        blc_mean = pickle.load(f)
    branch = '_highISO' if iso > 1600 else '_lowISO'
    ds_k = np.load(os.path.join('./resources', 'darkshading%s_k.npy' % branch), allow_pickle=True)
    ds_b = np.load(os.path.join('./resources', 'darkshading%s_b.npy' % branch), allow_pickle=True)
    darkshading[iso] = ds_k * iso + ds_b + blc_mean[iso]
    return darkshading[iso]


def pack_raw_SID_dark_shading(raw, iso):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern == 0)
    G1 = np.where(raw_pattern == 1)
    B = np.where(raw_pattern == 2)
    G2 = np.where(raw_pattern == 3)

    im = im - get_darkshading(iso)

    white_point = raw.white_level
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[R[0][0]:H:2, R[1][0]:W:2],  # RGBG
                    im[G1[0][0]:H:2, G1[1][0]:W:2],
                    im[B[0][0]:H:2, B[1][0]:W:2],
                    im[G2[0][0]:H:2, G2[1][0]:W:2]), axis=0).astype(np.float32)

    black_level = np.array(raw.black_level_per_channel)[:, None, None].astype(np.float32)

    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)

    return out


class ELDEvalDataset(data.Dataset):
    def __init__(self, basedir, camera_suffix, scenes=None, img_ids=None):
        super(ELDEvalDataset, self).__init__()
        self.basedir = basedir
        self.camera_suffix = camera_suffix  # ('Canon', '.CR2')
        self.scenes = scenes
        self.img_ids = img_ids

    def __getitem__(self, i):
        camera, suffix = self.camera_suffix

        scene_id = i // len(self.img_ids)
        img_id = i % len(self.img_ids)

        scene = 'scene-{}'.format(self.scenes[scene_id])

        datadir = join(self.basedir, camera, scene)

        input_path = join(datadir, 'IMG_{:04d}{}'.format(self.img_ids[img_id], suffix))

        gt_ids = np.array([1, 6, 11, 16])
        ind = np.argmin(np.abs(self.img_ids[img_id] - gt_ids))

        target_path = join(datadir, 'IMG_{:04d}{}'.format(gt_ids[ind], suffix))

        target_iso, target_expo = metainfo_ELD(target_path)
        target_expo = target_iso * target_expo
        iso, expo = metainfo_ELD(input_path)

        ratio = target_expo / (iso * expo)

        with rawpy.imread(input_path) as raw:
            # input = pack_raw_SID(raw) * ratio
            input = pack_raw_SID_dark_shading(raw, iso) * ratio

        with rawpy.imread(target_path) as raw:
            target = pack_raw_SID(raw)

        input = np.maximum(np.minimum(input, 1.0), 0)
        target = np.maximum(np.minimum(target, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)

        data = {'input': input, 'target': target, 'fn': input_path, 'iso': target_iso, 'rawpath': target_path, 'ratio': ratio}

        return data

    def __len__(self):
        return len(self.scenes) * len(self.img_ids)


class NPYDataset_SID(data.Dataset):
    def __init__(self, root, image_list_file, mode=None, patch_size=None):
        super(NPYDataset_SID, self).__init__()
        assert os.path.exists(root), "root: {} not found.".format(root)
        self.root = root
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file
        self.mode = mode
        self.patch_size = patch_size
        self.channel = 4
        self.img_info = []
        if self.mode == "train":
            self.train_image_filenames_clean = sorted(glob.glob(os.path.join(self.root, 'train_clean', '*.npy')))
            self.train_image_filenames_noisy = sorted(glob.glob(os.path.join(self.root, 'train_noisy', '*.npy')))
        elif self.mode == "test":
            with open(self.image_list_file, 'r') as f:
                for i, img_pair in enumerate(f):
                    img_pair = img_pair.strip()  # ./Sony/short/10003_00_0.04s.ARW ./Sony/long/10003_00_10s.ARW ISO200 F9
                    img_file, lbl_file, iso, focus = img_pair.split(' ')
                    ISO = int(''.join([x for x in iso if x.isdigit()]))
                    img_exposure = float(os.path.split(img_file)[-1][9:-5])  # 0.04
                    lbl_exposure = float(os.path.split(lbl_file)[-1][9:-5])  # 10
                    ratio = min(lbl_exposure / img_exposure, 300)
                    if ratio == 300:
                        self.img_info.append({
                            'img': img_file,
                            'lbl': lbl_file,
                            'img_exposure': img_exposure,
                            'lbl_exposure': lbl_exposure,
                            'ratio': ratio,
                            'iso': ISO,
                            'focus': focus,
                        })

    def __getitem__(self, index):
        if self.mode == "train":
            clean_image = np.load(self.train_image_filenames_clean[index], mmap_mode='r')
            noisy_image = np.load(self.train_image_filenames_noisy[index], mmap_mode='r')

            # Data Augmentations
            clean_full = tf.to_tensor(clean_image.transpose(1, 2, 0).copy())
            noisy_full = tf.to_tensor(noisy_image.transpose(1, 2, 0).copy())

            H, W = clean_full.shape[1], clean_full.shape[2]
            xx = np.random.randint(0, W - self.patch_size)
            yy = np.random.randint(0, H - self.patch_size)
            clean_full = clean_full[:, yy:yy + self.patch_size, xx:xx + self.patch_size]
            noisy_full = noisy_full[:, yy:yy + self.patch_size, xx:xx + self.patch_size]

            # Data Augmentations
            aug = random.randint(0, 8)
            if aug == 1:
                clean_full = clean_full.flip(1)
                noisy_full = noisy_full.flip(1)
            elif aug == 2:
                clean_full = clean_full.flip(2)
                noisy_full = noisy_full.flip(2)
            elif aug == 3:
                clean_full = torch.rot90(clean_full, dims=(1, 2))
                noisy_full = torch.rot90(noisy_full, dims=(1, 2))
            elif aug == 4:
                clean_full = torch.rot90(clean_full, dims=(1, 2), k=2)
                noisy_full = torch.rot90(noisy_full, dims=(1, 2), k=2)
            elif aug == 5:
                clean_full = torch.rot90(clean_full, dims=(1, 2), k=3)
                noisy_full = torch.rot90(noisy_full, dims=(1, 2), k=3)
            elif aug == 6:
                clean_full = torch.rot90(clean_full.flip(1), dims=(1, 2))
                noisy_full = torch.rot90(noisy_full.flip(1), dims=(1, 2))
            elif aug == 7:
                clean_full = torch.rot90(clean_full.flip(2), dims=(1, 2))
                noisy_full = torch.rot90(noisy_full.flip(2), dims=(1, 2))

            dic = {'clean': clean_full, 'noisy': noisy_full}
            return dic

        elif self.mode == "test":
            info = self.img_info[index]
            noisy_fn, clean_fn, ratio, iso = info['img'], info['lbl'], info['ratio'], info['iso']
            clean_path = os.path.join(self.root, clean_fn)
            noisy_path = os.path.join(self.root, noisy_fn)
            with rawpy.imread(clean_path) as raw_clean:
                clean_image = pack_raw_SID(raw_clean)
            with rawpy.imread(noisy_path) as raw_noisy:
                # noisy_image = pack_raw_bayer(raw_noisy) * ratio
                noisy_image = pack_raw_SID_dark_shading(raw_noisy, iso) * ratio

            noisy_full = np.maximum(np.minimum(noisy_image, 1.0), 0)
            noisy_full = np.ascontiguousarray(noisy_full)
            clean_full = np.ascontiguousarray(clean_image)

            clean_full = tf.to_tensor(clean_full.transpose(1, 2, 0).copy())
            noisy_full = tf.to_tensor(noisy_full.transpose(1, 2, 0).copy())

            dic = {'clean': clean_full, 'noisy': noisy_full, 'clean_path': clean_path}
            return dic

    def __len__(self):
        if self.mode == "train":
            return len(self.train_image_filenames_clean)
        elif self.mode == "test":
            return len(self.img_info)


class NPYDataset_LRD(data.Dataset):
    def __init__(self, root, image_list_file, mode=None, patch_size=None, batch_size=None, crop=True):
        super(NPYDataset_LRD, self).__init__()
        self.mode = mode
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.root = root
        self.crop = crop
        self.image_list_file = image_list_file
        self.img_info = []
        if self.mode == "train":
            self.train_image_filenames_clean = sorted(glob.glob(os.path.join(self.root, 'train_clean', '*.npy')))
            self.train_image_filenames_noisy = sorted(glob.glob(os.path.join(self.root, 'train_noisy', '*.npy')))
        elif self.mode == "test":
            with open(self.image_list_file, 'r') as f:
                for i, img_pair in enumerate(f):
                    img_pair = img_pair.strip()
                    clean_file, noisy_file, iso, ev, _ = img_pair.split(' ')
                    ISO = int(''.join([x for x in iso if x.isdigit()]))
                    EV = -float(''.join([x for x in ev if x.isdigit()]))
                    clean_exposure = float(os.path.split(clean_file)[-1][9:-5])
                    noisy_exposure = float(os.path.split(noisy_file)[-1][9:-5])
                    ratio = (100 * clean_exposure) / (ISO * noisy_exposure)
                    if -3.5 < EV < -2.5:
                        self.img_info.append({
                            'clean': clean_file,
                            'noisy': noisy_file,
                            'clean_exposure': clean_exposure,
                            'noisy_exposure': noisy_exposure,
                            'ratio': ratio,
                            'iso': ISO,
                            'ev': EV,
                        })

    def __getitem__(self, index):
        if self.mode == "train":
            clean_raw = np.load(self.train_image_filenames_clean[index], mmap_mode='r')
            noisy_raw = np.load(self.train_image_filenames_noisy[index], mmap_mode='r')

            H = clean_raw.shape[1]
            W = clean_raw.shape[2]

            ps = self.patch_size

            xx = np.random.randint(0, W - ps)
            yy = np.random.randint(0, H - ps)

            noisy_raw = noisy_raw[:, yy:yy + ps, xx:xx + ps]
            clean_raw = clean_raw[:, yy:yy + ps, xx:xx + ps]

            # Data Augmentations
            clean_full = clean_raw
            noisy_full = noisy_raw

            if np.random.randint(2, size=1)[0] == 1:  # random flip
                clean_full = np.flip(clean_full, axis=1)  # H
                noisy_full = np.flip(noisy_full, axis=1)
            if np.random.randint(2, size=1)[0] == 1:
                clean_full = np.flip(clean_full, axis=2)  # W
                noisy_full = np.flip(noisy_full, axis=2)
            if np.random.randint(2, size=1)[0] == 1:  # random transpose
                clean_full = np.transpose(clean_full, (0, 2, 1))
                noisy_full = np.transpose(noisy_full, (0, 2, 1))

            noisy_full = np.maximum(np.minimum(noisy_full, 1.0), 0)
            noisy_full = np.ascontiguousarray(noisy_full)
            clean_full = np.ascontiguousarray(clean_full)

            clean_full = tf.to_tensor(clean_full.transpose(1, 2, 0).copy())
            noisy_full = tf.to_tensor(noisy_full.transpose(1, 2, 0).copy())

            dic = {'clean': clean_full, 'noisy': noisy_full}
            return dic

        elif self.mode == "test":
            info = self.img_info[index]
            clean_fn, noisy_fn, ratio, iso = info['clean'], info['noisy'], info['ratio'], info['iso']
            clean_path = os.path.join(self.root, clean_fn)
            noisy_path = os.path.join(self.root, noisy_fn)
            raw_clean = rawpy.imread(clean_path)
            clean_raw = pack_raw_LRD(raw_clean)
            raw_noisy = rawpy.imread(noisy_path)
            noisy_raw = pack_raw_LRD(raw_noisy) * ratio

            noisy_full = np.maximum(np.minimum(noisy_raw, 1.0), 0)
            noisy_full = np.ascontiguousarray(noisy_full)
            clean_full = np.ascontiguousarray(clean_raw)

            clean_full = tf.to_tensor(clean_full.transpose(1, 2, 0).copy())
            noisy_full = tf.to_tensor(noisy_full.transpose(1, 2, 0).copy())

            dic = {'clean': clean_full, 'noisy': noisy_full, 'clean_path': clean_path}
            return dic

    def __len__(self):
        if self.mode == "train":
            return len(self.train_image_filenames_clean)
        elif self.mode == "test":
            return len(self.img_info)
