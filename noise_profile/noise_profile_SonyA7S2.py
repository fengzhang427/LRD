import torch
import numpy as np
import time
import rawpy
import torch.distributions as tdist
from scipy import stats
from raw_process.SID_post_process import load_CRF, raw2rgb_postprocess
from utils import write_images

Dual_ISO_Cameras = ['SonyA7S2']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pack_raw_DJI(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern == 0)
    G1 = np.where(raw_pattern == 1)
    B = np.where(raw_pattern == 2)
    G2 = np.where(raw_pattern == 3)

    white_point = 65535.0

    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[0:H:2, 0:W:2],  # RGGB
                    im[0:H:2, 1:W:2],
                    im[1:H:2, 0:W:2],
                    im[1:H:2, 1:W:2]), axis=0).astype(np.float32)

    # out = np.stack((im[R[0][0]:H:2, R[1][0]:W:2],  # RGGB
    #                 im[G1[0][0]:H:2, G1[1][0]:W:2],
    #                 im[G2[0][0]:H:2, G2[1][0]:W:2],
    #                 im[B[0][0]:H:2, B[1][0]:W:2]), axis=0).astype(np.float32)

    black_level = np.array(raw.black_level_per_channel)[:, None, None].astype(np.float32)

    # if max(raw.black_level_per_channel) != min(raw.black_level_per_channel):
    #     black_level = 2**round(np.log2(np.max(black_level)))
    # print(black_level)

    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0, 1)
    return out


def pack_raw_bayer(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern == 0)
    G1 = np.where(raw_pattern == 1)
    B = np.where(raw_pattern == 2)
    G2 = np.where(raw_pattern == 3)

    white_point = 16383.0
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


def log(string, log=None, str=False, end='\n', notime=False):
    log_string = f'{time.strftime("%Y-%m-%d %H:%M:%S")} >>  {string}' if not notime else string
    print(log_string)
    if log is not None:
        with open(log, 'a+') as f:
            f.write(log_string + '\n')
    else:
        pass
        # os.makedirs('worklog', exist_ok=True)
        # log = f'worklog/worklog-{time.strftime("%Y-%m-%d")}.txt'
        # with open(log,'a+') as f:
        #     f.write(log_string+'\n')
    if str:
        return string + end


def get_camera_noisy_params(camera_type=None):
    cam_noisy_params = {'NikonD850': {
        'Kmin': 1.2, 'Kmax': 2.4828, 'lam': -0.26, 'q': 1 / (2 ** 14), 'wp': 16383, 'bl': 512,
        'sigTLk': 0.906, 'sigTLb': -0.6754, 'sigTLsig': 0.035165,
        'sigRk': 0.8322, 'sigRb': -2.3326, 'sigRsig': 0.301333,
        'sigGsk': 0.8322, 'sigGsb': -0.1754, 'sigGssig': 0.035165,
    }, 'IMX686': {  # ISO-640~6400
        'Kmin': -0.19118, 'Kmax': 2.16820, 'lam': 0.102, 'q': 1 / (2 ** 10), 'wp': 1023, 'bl': 64,
        'sigTLk': 0.85187, 'sigTLb': 0.07991, 'sigTLsig': 0.02921,
        'sigRk': 0.87611, 'sigRb': -2.11455, 'sigRsig': 0.03274,
        'sigGsk': 0.85187, 'sigGsb': 0.67991, 'sigGssig': 0.02921,
    }, 'SonyA7S2_lowISO': {
        'Kmin': -1.67214, 'Kmax': 0.42228, 'lam': -0.026, 'q': 1 / (2 ** 14), 'wp': 16383, 'bl': 512,
        'sigRk': 0.78782, 'sigRb': -0.34227, 'sigRsig': 0.02832,
        'sigTLk': 0.74043, 'sigTLb': 0.86182, 'sigTLsig': 0.00712,
        'sigGsk': 0.82966, 'sigGsb': 1.49343, 'sigGssig': 0.00359,
        'sigReadk': 0.82879, 'sigReadb': 1.50601, 'sigReadsig': 0.00362,
        'uReadk': 0.01472, 'uReadb': 0.01129, 'uReadsig': 0.00034,
    }, 'SonyA7S2_highISO': {
        'Kmin': 0.64567, 'Kmax': 2.51606, 'lam': -0.025, 'q': 1 / (2 ** 14), 'wp': 16383, 'bl': 512,
        'sigRk': 0.62945, 'sigRb': -1.51040, 'sigRsig': 0.02609,
        'sigTLk': 0.74901, 'sigTLb': -0.12348, 'sigTLsig': 0.00638,
        'sigGsk': 0.82878, 'sigGsb': 0.44162, 'sigGssig': 0.00153,
        'sigReadk': 0.82645, 'sigReadb': 0.45061, 'sigReadsig': 0.00156,
        'uReadk': 0.00385, 'uReadb': 0.00674, 'uReadsig': 0.00039,
    }, 'CRVD': {
        'Kmin': 1.31339, 'Kmax': 3.95448, 'lam': 0.015, 'q': 1 / (2 ** 12), 'wp': 4095, 'bl': 240,
        'sigRk': 0.93368, 'sigRb': -2.19692, 'sigRsig': 0.02473,
        'sigGsk': 0.95387, 'sigGsb': 0.01552, 'sigGssig': 0.00855,
        'sigTLk': 0.95495, 'sigTLb': 0.01618, 'sigTLsig': 0.00790
    }}
    if camera_type in cam_noisy_params:
        return cam_noisy_params[camera_type]
    else:
        log(f'''Warning: we have not test the noisy parameters of camera "{camera_type}". Now we use NikonD850's parameters to test.''')
        return cam_noisy_params['NikonD850']


def get_camera_noisy_params_max(camera_type=None):
    cam_noisy_params = {'SonyA7S2_50': {'Kmax': 0.047815, 'lam': 0.1474653, 'sigGs': 1.0164667, 'sigGssig': 0.005272454,
                                        'sigTL': 0.70727646, 'sigTLsig': 0.004360543, 'sigR': 0.13997398,
                                        'sigRsig': 0.0064381803, 'bias': 0, 'biassig': 0.010093017,
                                        'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_64': {'Kmax': 0.0612032, 'lam': 0.13243394, 'sigGs': 1.0509665,
                                        'sigGssig': 0.008081373, 'sigTL': 0.71535635, 'sigTLsig': 0.0056863446,
                                        'sigR': 0.14346549, 'sigRsig': 0.006400559, 'bias': 0, 'biassig': 0.008690166,
                                        'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_80': {'Kmax': 0.076504, 'lam': 0.1121489, 'sigGs': 1.180899, 'sigGssig': 0.011333668,
                                        'sigTL': 0.7799473, 'sigTLsig': 0.009347968, 'sigR': 0.19540153,
                                        'sigRsig': 0.008197397, 'bias': 0, 'biassig': 0.0107246125,
                                        'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_100': {'Kmax': 0.09563, 'lam': 0.14875287, 'sigGs': 1.0067395,
                                         'sigGssig': 0.0033682834, 'sigTL': 0.70181876, 'sigTLsig': 0.0037532174,
                                         'sigR': 0.1391465, 'sigRsig': 0.006530218, 'bias': 0, 'biassig': 0.007235429,
                                         'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_125': {'Kmax': 0.1195375, 'lam': 0.12904578, 'sigGs': 1.0279676,
                                         'sigGssig': 0.007364685, 'sigTL': 0.6961967, 'sigTLsig': 0.0048687346,
                                         'sigR': 0.14485553, 'sigRsig': 0.006731584, 'bias': 0, 'biassig': 0.008026363,
                                         'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_160': {'Kmax': 0.153008, 'lam': 0.094135, 'sigGs': 1.1293099, 'sigGssig': 0.008340453,
                                         'sigTL': 0.7258587, 'sigTLsig': 0.008032158, 'sigR': 0.19755602,
                                         'sigRsig': 0.0082754735, 'bias': 0, 'biassig': 0.0101351, 'q': 6.103515625e-05,
                                         'wp': 16383, 'bl': 512},
                        'SonyA7S2_200': {'Kmax': 0.19126, 'lam': 0.07902429, 'sigGs': 1.2926387,
                                         'sigGssig': 0.012171176, 'sigTL': 0.8117464, 'sigTLsig': 0.010250768,
                                         'sigR': 0.22815849, 'sigRsig': 0.010726711, 'bias': 0, 'biassig': 0.011413908,
                                         'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_250': {'Kmax': 0.239075, 'lam': 0.051688068, 'sigGs': 1.4345995,
                                         'sigGssig': 0.01606571, 'sigTL': 0.8630922, 'sigTLsig': 0.013844714,
                                         'sigR': 0.26271912, 'sigRsig': 0.0130637, 'bias': 0, 'biassig': 0.013569083,
                                         'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_320': {'Kmax': 0.306016, 'lam': 0.040700804, 'sigGs': 1.7481371,
                                         'sigGssig': 0.019626873, 'sigTL': 1.0334468, 'sigTLsig': 0.017629284,
                                         'sigR': 0.3097104, 'sigRsig': 0.016202712, 'bias': 0, 'biassig': 0.017825918,
                                         'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_400': {'Kmax': 0.38252, 'lam': 0.0222538, 'sigGs': 2.0595572, 'sigGssig': 0.024872316,
                                         'sigTL': 1.1816813, 'sigTLsig': 0.02505812, 'sigR': 0.36209714,
                                         'sigRsig': 0.01994737, 'bias': 0, 'biassig': 0.021005306, 'q': 6.103515625e-05,
                                         'wp': 16383, 'bl': 512},
                        'SonyA7S2_500': {'Kmax': 0.47815, 'lam': -0.0031342343, 'sigGs': 2.3956928,
                                         'sigGssig': 0.030144656, 'sigTL': 1.31772, 'sigTLsig': 0.028629242,
                                         'sigR': 0.42528257, 'sigRsig': 0.025104137, 'bias': 0, 'biassig': 0.02981831,
                                         'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_640': {'Kmax': 0.612032, 'lam': 0.002566592, 'sigGs': 2.9662898,
                                         'sigGssig': 0.045661453, 'sigTL': 1.6474211, 'sigTLsig': 0.04671843,
                                         'sigR': 0.48839623, 'sigRsig': 0.031589635, 'bias': 0, 'biassig': 0.10000693,
                                         'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_800': {'Kmax': 0.76504, 'lam': -0.008199721, 'sigGs': 3.5475867,
                                         'sigGssig': 0.052318197, 'sigTL': 1.9346539, 'sigTLsig': 0.046128694,
                                         'sigR': 0.5723769, 'sigRsig': 0.037824076, 'bias': 0, 'biassig': 0.025339302,
                                         'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_1000': {'Kmax': 0.9563, 'lam': -0.021061005, 'sigGs': 4.2727833,
                                          'sigGssig': 0.06972333, 'sigTL': 2.2795107, 'sigTLsig': 0.059203167,
                                          'sigR': 0.6845563, 'sigRsig': 0.04879781, 'bias': 0, 'biassig': 0.027911892,
                                          'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_1250': {'Kmax': 1.195375, 'lam': -0.032423194, 'sigGs': 5.177596,
                                          'sigGssig': 0.092677385, 'sigTL': 2.708437, 'sigTLsig': 0.07622563,
                                          'sigR': 0.8177013, 'sigRsig': 0.06162229, 'bias': 0, 'biassig': 0.03293372,
                                          'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_1600': {'Kmax': 1.53008, 'lam': -0.0441045, 'sigGs': 6.29925, 'sigGssig': 0.1153261,
                                          'sigTL': 3.2283993, 'sigTLsig': 0.09118158, 'sigR': 0.988786,
                                          'sigRsig': 0.078567736, 'bias': 0, 'biassig': 0.03877672,
                                          'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_2000': {'Kmax': 1.9126, 'lam': -0.012963797, 'sigGs': 2.653871,
                                          'sigGssig': 0.015890995, 'sigTL': 1.4356787, 'sigTLsig': 0.02178686,
                                          'sigR': 0.33124214, 'sigRsig': 0.018801652, 'bias': 0, 'biassig': 0.01570677,
                                          'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_2500': {'Kmax': 2.39075, 'lam': -0.027097283, 'sigGs': 3.200225,
                                          'sigGssig': 0.019307792, 'sigTL': 1.6897862, 'sigTLsig': 0.025873765,
                                          'sigR': 0.38264316, 'sigRsig': 0.023769397, 'bias': 0, 'biassig': 0.018728448,
                                          'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_3200': {'Kmax': 3.06016, 'lam': -0.034863412, 'sigGs': 3.9193838,
                                          'sigGssig': 0.02649232, 'sigTL': 2.0417721, 'sigTLsig': 0.032873377,
                                          'sigR': 0.44543457, 'sigRsig': 0.030114045, 'bias': 0, 'biassig': 0.021355819,
                                          'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_4000': {'Kmax': 3.8252, 'lam': -0.043700505, 'sigGs': 4.8015847,
                                          'sigGssig': 0.03781628, 'sigTL': 2.4629273, 'sigTLsig': 0.042401053,
                                          'sigR': 0.52347374, 'sigRsig': 0.03929801, 'bias': 0, 'biassig': 0.026152484,
                                          'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_5000': {'Kmax': 4.7815, 'lam': -0.053150143, 'sigGs': 5.8995814,
                                          'sigGssig': 0.0625814, 'sigTL': 2.9761007, 'sigTLsig': 0.061326735,
                                          'sigR': 0.6190265, 'sigRsig': 0.05335372, 'bias': 0, 'biassig': 0.058574405,
                                          'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_6400': {'Kmax': 6.12032, 'lam': -0.07517104, 'sigGs': 7.1163535,
                                          'sigGssig': 0.08435366, 'sigTL': 3.4502964, 'sigTLsig': 0.08226275,
                                          'sigR': 0.7218788, 'sigRsig': 0.0642334, 'bias': 0, 'biassig': 0.059074216,
                                          'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_8000': {'Kmax': 7.6504, 'lam': -0.08208357, 'sigGs': 8.916516, 'sigGssig': 0.12763213,
                                          'sigTL': 4.269624, 'sigTLsig': 0.13381928, 'sigR': 0.87760293,
                                          'sigRsig': 0.07389065, 'bias': 0, 'biassig': 0.084842026,
                                          'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_10000': {'Kmax': 9.563, 'lam': -0.073289566, 'sigGs': 11.291476,
                                           'sigGssig': 0.1639773, 'sigTL': 5.495318, 'sigTLsig': 0.16279395,
                                           'sigR': 1.0522343, 'sigRsig': 0.094359785, 'bias': 0, 'biassig': 0.107438326,
                                           'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_12800': {'Kmax': 12.24064, 'lam': -0.06495205, 'sigGs': 14.245901,
                                           'sigGssig': 0.17283991, 'sigTL': 7.038261, 'sigTLsig': 0.18822834,
                                           'sigR': 1.2749791, 'sigRsig': 0.120479785, 'bias': 0, 'biassig': 0.0944684,
                                           'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_16000': {'Kmax': 15.3008, 'lam': -0.060692135, 'sigGs': 17.833515,
                                           'sigGssig': 0.19809262, 'sigTL': 8.877547, 'sigTLsig': 0.23338738,
                                           'sigR': 1.5559287, 'sigRsig': 0.15791349, 'bias': 0, 'biassig': 0.09725099,
                                           'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_20000': {'Kmax': 19.126, 'lam': -0.060213074, 'sigGs': 22.084776,
                                           'sigGssig': 0.21820943, 'sigTL': 11.002351, 'sigTLsig': 0.28806436,
                                           'sigR': 1.8810822, 'sigRsig': 0.18937257, 'bias': 0, 'biassig': 0.4984733,
                                           'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
                        'SonyA7S2_25600': {'Kmax': 24.48128, 'lam': -0.09089118, 'sigGs': 25.853043,
                                           'sigGssig': 0.35371417, 'sigTL': 12.175712, 'sigTLsig': 0.4215717,
                                           'sigR': 2.2760193, 'sigRsig': 0.2609267, 'bias': 0, 'biassig': 0.37568903,
                                           'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 'IMX686_6400': {
            'Kmax': 8.74253, 'sigGs': 12.8901, 'sigGssig': 0.03,
            'sigTL': 12.8901, 'lam': 0.015, 'sigR': 0,
            'q': 1 / (2 ** 10), 'wp': 1023, 'bl': 64, 'bias': -0.56896687
        }}
    if camera_type in cam_noisy_params:
        return cam_noisy_params[camera_type]
    else:
        # log(f'''Warning: we have not test the noisy parameters of camera "{camera_type}".''')
        return None


def sample_params_max(camera_type='NikonD850', ratio=None, iso=None):
    # 获取已经测算好的相机噪声参数
    params = None
    if iso is not None:
        camera_type_iso = camera_type + f'_{iso}'
        params = get_camera_noisy_params_max(camera_type=camera_type_iso)
    if params is None:
        if camera_type in Dual_ISO_Cameras:
            choice = np.random.randint(2)
            camera_type += '_lowISO' if choice < 1 else '_highISO'
        params = get_camera_noisy_params(camera_type=camera_type)
        # 根据最小二乘法得到的噪声参数回归模型采样噪声参数
        bias = 0
        log_K = params['Kmax'] + np.random.uniform(low=-0.01, high=+0.01)  # 增加一些扰动，以防测的不准
        K = np.exp(log_K)
        mu_TL = params['sigTLk'] * log_K + params['sigTLb']
        mu_R = params['sigRk'] * log_K + params['sigRb']
        mu_Gs = params['sigGsk'] * log_K + params['sigGsb'] if 'sigGsk' in params else 2 ** (-14)
        # 去掉log
        sigTL = np.exp(mu_TL)
        sigR = np.exp(mu_R)
        sigGs = np.exp(np.random.normal(loc=mu_Gs, scale=params['sigGssig']) if 'sigGssig' in params else mu_Gs)
    else:
        K = params['Kmax'] * (1 + np.random.uniform(low=-0.01, high=+0.01))  # 增加一些扰动，以防测的不准
        sigGs = np.random.normal(loc=params['sigGs'], scale=params['sigGssig']) if 'sigGssig' in params else params[
            'sigGs']
        sigTL = np.random.normal(loc=params['sigTL'], scale=params['sigTLsig']) if 'sigTLsig' in params else params[
            'sigTL']
        sigR = np.random.normal(loc=params['sigR'], scale=params['sigRsig']) if 'sigRsig' in params else params['sigR']
        bias = params['bias']
    wp = params['wp']
    bl = params['bl']
    lam = params['lam']
    q = params['q']

    if ratio is None:
        if 'SonyA7S2' in camera_type:
            ratio = np.random.uniform(low=100, high=300)
        else:
            log_ratio = np.random.uniform(low=0, high=2.08)
            ratio = np.exp(log_ratio)

    return {'K': K, 'sigTL': sigTL, 'sigR': sigR, 'sigGs': sigGs, 'bias': bias,
            'lam': lam, 'q': q, 'ratio': ratio, 'wp': wp, 'bl': bl}


def generate_noisy_obs(y, camera_type=None, wp=16383, noise_code='p', param=None, MultiFrameMean=1, ori=False,
                       clip=False):
    # # Burst denoising
    # sig_read = 10. ** np.random.uniform(low=-3., high=-1.5)
    # sig_shot = 10. ** np.random.uniform(low=-2., high=-1.)
    # shot = np.random.randn(*y.shape).astype(np.float32) * np.sqrt(np.maximum(y, 1e-10)) * sig_shot
    # read = np.random.randn(*y.shape).astype(np.float32) * sig_read
    # z = y + shot + read
    p = param
    y = y * (p['wp'] - p['bl'])
    # p['ratio'] = 1/p['ratio'] # 临时行为，为了快速实现MFM
    y = y / p['ratio']
    MFM = MultiFrameMean ** 0.5

    use_R = True if 'r' in noise_code.lower() else False
    use_Q = True if 'q' in noise_code.lower() else False
    use_TL = True if 'g' in noise_code.lower() else False
    use_P = True if 'p' in noise_code.lower() else False
    use_D = True if 'd' in noise_code.lower() else False
    use_black = True if 'b' in noise_code.lower() else False

    if use_P:  # 使用泊松噪声作为shot noisy
        noisy_shot = np.random.poisson(MFM * y / p['K']).astype(np.float32) * p['K'] / MFM
    else:  # 不考虑shot noisy
        noisy_shot = y + np.random.randn(*y.shape).astype(np.float32) * np.sqrt(np.maximum(y / p['K'], 1e-10)) * p[
            'K'] / MFM
    if not use_black:
        if use_TL:  # 使用TL噪声作为read noisy
            noisy_read = stats.tukeylambda.rvs(p['lam'], scale=p['sigTL'] / MFM, size=y.shape).astype(np.float32)
        else:  # 使用高斯噪声作为read noisy
            noisy_read = stats.norm.rvs(scale=p['sigGs'] / MFM, size=y.shape).astype(np.float32)
        # 行噪声需要使用行的维度h，[1,c,h,w]所以-2是h
        noisy_row = np.random.randn(y.shape[-3], y.shape[-2], 1).astype(np.float32) * p['sigR'] / MFM if use_R else 0
        noisy_q = np.random.uniform(low=-0.5, high=0.5, size=y.shape) if use_Q else 0
        noisy_bias = p['bias'] if use_D else 0
    else:
        noisy_read = 0
        noisy_row = 0
        noisy_q = 0
        noisy_bias = 0

    # 归一化回[0, 1]
    z = (noisy_shot + noisy_read + noisy_row + noisy_q + noisy_bias) / (p['wp'] - p['bl'])
    # 模拟实际情况
    z = np.clip(z, -p['bl'] / p['wp'], 1) if clip is False else np.clip(z, 0, 1)
    if ori is False:
        z = z * p['ratio']

    return z.astype(np.float32)


def generate_noisy_torch(y, camera_type=None,  noise_code='p', param=None, MultiFrameMean=1, ori=False, clip=False):
    p = param
    y = y * (p['wp'] - p['bl'])
    # p['ratio'] = 1/p['ratio'] # 临时行为，为了快速实现MFM
    y = y / p['ratio']
    MFM = MultiFrameMean ** 0.5

    use_R = True if 'r' in noise_code.lower() else False
    use_Q = True if 'q' in noise_code.lower() else False
    use_TL = True if 'g' in noise_code.lower() else False
    use_P = True if 'p' in noise_code.lower() else False
    use_D = True if 'd' in noise_code.lower() else False
    use_black = True if 'b' in noise_code.lower() else False

    if use_P:   # 使用泊松噪声作为shot noisy
        noisy_shot = tdist.Poisson(MFM*y/p['K']).sample() * p['K'] / MFM
    else:   # 不考虑shot noisy
        noisy_shot = tdist.Normal(y).sample() * torch.sqrt(torch.max(y/p['K'], 1e-10)) * p['K'] / MFM
    if not use_black:
        if use_TL:   # 使用TL噪声作为read noisy
            raise NotImplementedError
            # noisy_read = stats.tukeylambda.rvs(p['lam'], scale=p['sigTL'], size=y.shape).astype(np.float32)
        else:   # 使用高斯噪声作为read noisy
            noisy_read = tdist.Normal(loc=torch.zeros_like(y), scale=p['sigGs']/MFM).sample()
    else:
        noisy_read = 0
    # 使用行噪声
    noisy_row = torch.randn(y.shape[-3], y.shape[-2], 1, device=DEVICE) * p['sigR'] / MFM if use_R else 0
    noisy_q = (torch.rand(y.shape, device=DEVICE) - 0.5) * p['q'] * (p['wp'] - p['bl']) if use_Q else 0
    noisy_bias = p['bias'] if use_D else 0

    # 归一化回[0, 1]
    z = (noisy_shot + noisy_read + noisy_row + noisy_q + noisy_bias) / (p['wp'] - p['bl'])
    # 模拟实际情况
    z = torch.clamp(z, -p['bl']/p['wp'], 1) if clip is False else torch.clamp(z, 0, 1)
    # ori_brightness
    if ori is False:
        z = z * p['ratio']

    return z


if __name__ == '__main__':
    path = '/data/ELD/SonyA7S2/scene-8'
    path_clean = '/data/SID/Sony/long/00200_00_10s.ARW'
    path_noisy = '/data/SID/Sony/short/00200_00_0.1s.ARW'
    clean_raw = rawpy.imread(path_clean)
    noisy_raw = rawpy.imread(path_noisy)
    clean_img = pack_raw_bayer(clean_raw)
    noisy_img = pack_raw_bayer(noisy_raw) * 100.0

    p = sample_params_max(camera_type='SonyA7S2', ratio=100, iso=1600)
    noisy = generate_noisy_obs(clean_img, camera_type='SonyA7S2', noise_code='p', param=p)

    clean_tensor = torch.from_numpy(clean_img).unsqueeze(0)
    noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0)
    # fake_noisy_tensor = torch.from_numpy(noisy).unsqueeze(0)

    fake_noisy_tensor = generate_noisy_torch(clean_tensor, camera_type='SonyA7S2', noise_code='p', param=p)

    CRF = load_CRF()
    clean_image = raw2rgb_postprocess(clean_tensor, clean_raw, CRF)
    noisy_image = raw2rgb_postprocess(noisy_tensor, noisy_raw, CRF)
    fake_noisy_image = raw2rgb_postprocess(fake_noisy_tensor, clean_raw, CRF)

    sample = clean_image, noisy_image, fake_noisy_image
    write_images(sample, 3, '1_noisy_debug.bmp')
    print('test')
    print("")
