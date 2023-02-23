# Learning_AVSegmentation 👀
--Vessel segmentation, artery and vein, retinal image

--Code for MICCAI paper ["Learning to Address Intra-segment Misclassification in Retinal Imaging"](https://link.springer.com/chapter/10.1007/978-3-030-87193-2_46)

Please contact 	**ykzhoua@gmail.com** or **yukun.zhou.19@ucl.ac.uk** if you have questions.


## Brief Background

This repository aims at improving multi-class vessel segmentation performance in retinal fundus photograph by alleviating the intra-segment misclassification around the intersections. The research data sets in experiments include DRIVE-AV [<sup>1,2</sup>](#refer-anchor-1), LES-AV [<sup>3</sup>](#refer-anchor-2), and HRF-AV [<sup>4,5</sup>](#refer-anchor-3).

![image](./imgs/Figure1.jpg)

## Advantages

There are a few strengths in this work:

1. We strictly evaluate the method performance in multi-class segmentation manner, instead of only considering the classification accuracy (previous evaluation). Mean value and standard deviation are calculated to show robust performance in test.
2. The GAN-based segmentation backbone is revised based on a SOTA vessel segmentation method [<sup>6</sup>](#refer-anchor-4).
3. The binary-to-multi fusion network avoids directly learning on the ambiguous pixel label brought by intersections, achieving SOTA performance on multi-class vessel segmentation.
4.  The code and algorithm are easily transferred to other medical  or natural linear segmentation fields.



## Install

### Requirements

1. Work on Linux and Windows, but Linux is preferred to replicate the reported performance.
2. This project is based on pytorch==1.6.0, torchvision==0.7.0, CUDAToolkit==10.1(10.2-11.3 is capable).
3. A GPU is essential. In our work, we utilise one Tesla T4 with 15 GB of DRAM. If with weaker GPU, we suggest to change the image size setting in `scripts.utils.py`


Packages installation:
```
pip install -r requirements.txt
```

## Usage


### Pretrained Model

The pretrained model are provided in [Google_DRIVE](https://drive.google.com/drive/folders/1c_UZaq69RmPZjFvccot6GWqnhx2VzFRs?usp=share_link). Download them and unzip them directly at the project folder.


### Train

Start training, the dataset can be set as DRIVE_AV, LES-AV, or HRF-AV.
```
python train.py --e=500 \
                --batch-size=2 \
                --learning-rate=8e-4 \
                --v=10.0 \
                --alpha=0.5 \
                --beta=1.1 \
                --gama=0.08 \
                --dataset=DRIVE_AV \
                --discriminator=unet \
                --job_name=DRIVE_AV_randomseed_42 \
                --uniform=True \
                --seed_num=42
```

### Test
Test the trained models.
```
python test.py --batch-size=1 \
               --dataset=DRIVE_AV \
               --job_name=DRIVE_AV_randomseed \
               --uniform=True
```




 
## Performance

### Switch final activation map from sigmoid to softmax

| Test dataset  | Sensitivity        | AUC-ROC |    F1-score|  AUC-PR| MSE|
| ------------- | ------------------ |-------------|------------|-----|----|
| DRIVE-AV      | 70.8 &pm; 0.1     |     84.7 &pm; 0.05  |    71.99 &pm; 0.04        |  73.06 &pm; 0.03   |  2.85 &pm; 0.01  |
| LES-AV        | 64.41 &pm; 0.09  |     81.72 &pm; 0.04        |    67.22 &pm; 0.06        |  69.08 &pm; 0.06   |  2.22 &pm; 0.01  |
| HRF-AV        |       71.85 &pm; 0.29        |     85.38 &pm; 0.13        |       71.92 &pm; 0.03     |  73.23 &pm; 0.03   |  2 &pm; 0.01  |

&nbsp;
&nbsp;
<!--Code 
![image](./imgs/Figure2.jpg)
-->


## Reference 
<div id="refer-anchor-1"></div>

1) Staal J, Abràmoff M D, Niemeijer M, et al. Ridge-based vessel segmentation in color images of the retina[J]. IEEE transactions on medical imaging, 2004, 23(4): 501-509.

2) Hu Q, Abràmoff M D, Garvin M K. Automated separation of binary overlapping trees in low-contrast color retinal images[C]//International conference on medical image computing and computer-assisted intervention. Springer, Berlin, Heidelberg, 2013: 436-443.

3) Orlando J I, Breda J B, Van Keer K, et al. Towards a glaucoma risk index based on simulated hemodynamics from fundus images[C]//International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2018: 65-73.

4) Budai A, Bock R, Maier A, et al. Robust vessel segmentation in fundus images[J]. International journal of biomedical imaging, 2013, 2013.

5) Hemelings R, Elen B, Stalmans I, et al. Artery–vein segmentation in fundus images using a fully convolutional network[J]. Computerized Medical Imaging and Graphics, 2019, 76: 101636.

6) Zhou Y, Chen Z, Shen H, et al. A refined equilibrium generative adversarial network for retinal vessel segmentation[J]. Neurocomputing, 2021, 437: 118-130.


## Citation

```
@inproceedings{zhou2021learning,
  title={Learning to address intra-segment misclassification in retinal imaging},
  author={Zhou, Yukun and Xu, Moucheng and Hu, Yipeng and Lin, Hongxiang and Jacob, Joseph and Keane, Pearse A and Alexander, Daniel C},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2021: 24th International Conference, Strasbourg, France, September 27--October 1, 2021, Proceedings, Part I 24},
  pages={482--492},
  year={2021},
  organization={Springer}
}


@article{zhou2022automorph,
  title={AutoMorph: Automated Retinal Vascular Morphology Quantification Via a Deep Learning Pipeline},
  author={Zhou, Yukun and Wagner, Siegfried K and Chia, Mark A and Zhao, An and Xu, Moucheng and Struyven, Robbert and Alexander, Daniel C and Keane, Pearse A and others},
  journal={Translational vision science \& technology},
  volume={11},
  number={7},
  pages={12--12},
  year={2022},
  publisher={The Association for Research in Vision and Ophthalmology}
}
```


