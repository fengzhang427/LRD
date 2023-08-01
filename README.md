# Towards General Low-Light Raw Noise Synthesis and Modeling (ICCV 2023)
**Technical Report**: [![](https://img.shields.io/badge/Report-arXiv:2307.16508-green)](https://arxiv.org/abs/2307.16508)

<img width="900" alt="image" src='./asset/results.png'>

LRD is a general framework for low-light raw noise synthesis and modeling.
Specifically, we synthesize the signal-dependent and signal-independent noise in a physics- and learning-based manner, respectively. In this way, our method can be considered as a general model, that is, it can simultaneously learn different noise characteristics for different ISO levels and generalize to various sensors.

## :snake:Synthesis Pipeline
<img width="900" alt="image" src='./asset/framework.png'>

## :open_file_folder:LRD Datasets
Source code and datasets will be released soon.
<img width="900" alt="image" src='./asset/datasets.png'>

## :bookmark_tabs:Intallation
* Install the conda environment
```
conda create -n lrd python=3.8
conda activate lrd
```
* Install Pytorch
```commandline
conda install pytorch==1.9 torchvision cudatoolkit=10.2 -c pytorch
```
* Install Packages for Raw Image
```commandline
pip install rawpy
pip install ExifRead
pip install h5py
```
* Install other packages
```commandline
pip install tqdm
pip install lmdb
pip install glob
pip install imageio
pip install PyYAML
pip install timm
pip install patchify
conda install -c conda-forge scipy
pip install opencv-python
pip install tensorboardx
pip install scikit-image
pip install colour
pip install pylab-sdk
pip install pillow
```

## :car:Run
Source code and datasets will be released soon.

## :book: Citation
If you find our LRD model useful for you, please consider citing :mega:
```bibtex
@misc{zhang2023general,
      title={Towards General Low-Light Raw Noise Synthesis and Modeling}, 
      author={Feng Zhang and Bin Xu and Zhiqiang Li and Xinran Liu and Qingbo Lu and Changxin Gao and Nong Sang},
      year={2023},
      eprint={2307.16508},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
}
```

## :email:Contact
If you have any question, feel free to email fengzhangaia@hust.edu.cn.
