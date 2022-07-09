The implementation of [**Coarse-to-Fine Visual Place Recognition**](https://doi.org/10.1007/978-3-030-92273-3_3), ICONIP 2021.

### Abstract:

Visual Place Recognition still struggles against viewpoint change, confusion from similar patterns in different places, or high computation complexity. In this paper, we propose a progressive Coarse-To-Fine (CTF-VPR) framework, which has a strong ability on handling irrelevant matches and controlling time consumption.
It employs global descriptors to discover visually similar references and local descriptors to filter those with similar but irrelative patterns. Besides, a region-specific representing format called regional descriptor is introduced with region augmentation and increases the possibilities of positive references with partially relevant areas via region refinement. Furthermore, during the spatial verification, we provide the Spatial Deviation Index (SDI) considering coordinate deviation to evaluate the consistency of matches. It discards exhaustive and iterative search and reduces the time consumption hundreds of times.


### Codes in PyTorch:

The codes are heavily modified from:
+ [SFRS](https://github.com/yxgeee/OpenIBL): Self-supervising Fine-grained Region Similarities for Large-scale Image Localization, ECCV 2020.
+ [Patch-NetVLAD](https://github.com/QVPR/Patch-NetVLAD): Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition, CVPR 2021.

For dataset and pretrained models, please refer to SFRS.

#### Usage:

#### Global retrieval & Region refinement

```
xixi
```

|   Dataset   |  Recalls (Base) |   Recalls (RR)  |
|  :--------: | :-------------: | :-------------: |
|   Pitts250k   | 90.7 / 96.4 / 97.6 | 91.1 / 96.5 / 97.7 |
|   Tokyo24/7   | 84.1 / 89.2 / 91.3^\* | 87.0 / 92.4 / 93.7 |

^\*: This performance drop (85.4 / 91.1 / 93.3 in SFRS) is caused by smaller query size. Please refer to the 

#### Spatial verfication

```
xixi
```
|   Dataset   |   Recalls (RR)  |   Recalls (SV)  |
|  :--------: | :-------------: | :-------------: |
|   Pitts250k   | 91.1 / 96.5 / 97.7 | 92.6 / 97.2 / 97.7 |
|   Tokyo24/7   | 87.0 / 92.4 / 93.7 | 91.1 / 93.7 / 93.7 |

#### Time

This repo is a simple version of implementation neglecting the time cost.


### Citation:

If you find this code helpful for your research, please cite our paper.
```
@inproceedings{qi21ctfvpr,
  title = {Coarse-to-Fine Visual Place Recognition},
  author = {Junkun Qi and Rui Wang and Chuan Wang and Xiaochun Cao},
  booktitle = {International Conference on Neural Information Processing (ICONIP)},
  year = {2021}
}
```