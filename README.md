The implementation of [**Coarse-to-Fine Visual Place Recognition**](https://doi.org/10.1007/978-3-030-92273-3_3), ICONIP 2021.

### Abstract

Visual Place Recognition still struggles against viewpoint change, confusion from similar patterns in different places, or high computation complexity. In this paper, we propose a progressive Coarse-To-Fine (CTF-VPR) framework, which has a strong ability on handling irrelevant matches and controlling time consumption.
It employs global descriptors to discover visually similar references and local descriptors to filter those with similar but irrelative patterns. Besides, a region-specific representing format called regional descriptor is introduced with region augmentation and increases the possibilities of positive references with partially relevant areas via region refinement. Furthermore, during the spatial verification, we provide the Spatial Deviation Index (SDI) considering coordinate deviation to evaluate the consistency of matches. It discards exhaustive and iterative search and reduces the time consumption hundreds of times.


### Codes in PyTorch

The codes are heavily modified from:
+ [SFRS](https://github.com/yxgeee/OpenIBL): Self-supervising Fine-grained Region Similarities for Large-scale Image Localization, ECCV 2020.
+ [Patch-NetVLAD](https://github.com/QVPR/Patch-NetVLAD): Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition, CVPR 2021.

For the datasets and pretrained models, please refer to SFRS.

#### Usage

First, extract and save the image features.
+ For global and regional descriptors, run
```
python save_feat_region.py --data-dir <your_data_dir> --dataset <pitts_or_tokyo> --resume <pretrained_model_from_SFRS> --save-dir <your_dir_to_save_features>
```
+ For local descriptors, run
```
python save_feat_patch.py --data-dir <your_data_dir> --dataset <pitts_or_tokyo> --resume <pretrained_model_from_SFRS> --save-dir <your_dir_to_save_features>
```

Then, retrieve images from the database (gallery).
+ Global retrieval & Region refinement, run
```
python pred_region.py --data-dir <your_data_dir> --dataset <pitts_or_tokyo> --load-dir <your_dir_to_save_features> --save-dir <your_dir_to_save_predictions>
```
+ SDI-based spatial verification, run
```
python pred_patch.py --data-dir <your_data_dir> --dataset <pitts_or_tokyo> --load-dir <your_dir_to_save_features> --save-dir <your_dir_to_save_predictions>
```

The results will be the same as those claimed in the paper:
|   Dataset   |  Recalls (Base) |   Recalls (RR)  |   Recalls (SV)  |
|  :--------: | :-------------: | :-------------: |   Recalls (SV)  |
|   Pitts250k   | 90.7 / 96.4 / 97.6 | 91.1 / 96.5 / 97.7 | 92.6 / 97.2 / 97.7 |
|   Tokyo24/7   | 84.1 / 89.2 / 91.3\* | 87.0 / 92.4 / 93.7 | 91.1 / 93.7 / 93.7 |

\*: This performance drop (85.4 / 91.1 / 93.3 in SFRS) is caused by smaller query size. Please refer to datasets/\_\_init\_\_.py.

Note that this repo is a simple version neglecting the time cost.
For example, we use argsort func for global retrieval which is highly time-consuming.
Using [faiss](https://github.com/facebookresearch/faiss) can reduce the retrieval time.


### Citation

If you find this code helpful for your research, please cite our paper.
```
@inproceedings{qi21ctfvpr,
  title = {Coarse-to-Fine Visual Place Recognition},
  author = {Junkun Qi and Rui Wang and Chuan Wang and Xiaochun Cao},
  booktitle = {International Conference on Neural Information Processing (ICONIP)},
  year = {2021}
}
```