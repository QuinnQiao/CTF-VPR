import numpy as np
import torch
from tqdm.auto import tqdm
import os.path as osp
import time 


__all__ = ['local_matcher', 'local_matcher_tokyo']


def torch_nn(x, y):
    dist = torch.matmul(x, y)

    fw_inds = torch.argmax(dist, 0)
    bw_inds = torch.argmax(dist, 1)

    return fw_inds, bw_inds, dist[fw_inds, torch.arange(dist.size(1))]


class PatchMatcher(object):
    def __init__(self, patch_size, stride, keypoints):
        self.patch_size = patch_size
        self.stride = stride
        self.keypoints = keypoints

    def match(self, qfeat, dbfeat, query_keypoints=None):
        '''
        For Pittsburgh, the query and the gallery are of the fixed size (480*640).
        For Tokyo, the queries are of different sizes.
        Thus, we explicitly load the keypoints of queries.

        '''
        fw_inds, bw_inds, dists = torch_nn(qfeat, dbfeat)

        fw_inds = fw_inds.cpu().numpy()
        bw_inds = bw_inds.cpu().numpy()
        dists = dists.cpu().numpy()

        mutuals = np.atleast_1d(np.argwhere(bw_inds[fw_inds] == np.arange(len(fw_inds))).squeeze())
        selected = dists[mutuals] > 0.5
        mutuals = mutuals[selected]

        if len(mutuals) > 0:
            index_keypoints = self.keypoints[:, mutuals]
            if query_keypoints is None:
                query_keypoints = self.keypoints[:, fw_inds[mutuals]]
            else:
                query_keypoints = query_keypoints[:, fw_inds[mutuals]]

            spatial_dist = index_keypoints - query_keypoints 

            std = np.std(spatial_dist, axis=1).sum()
            inv = qfeat.shape[0] / len(mutuals)

            score = std * inv
        else:
            score = np.inf

        return score


def calc_receptive_boxes(height, width):
    """Calculate receptive boxes for each feature point.
    Modified from
    https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/delf/delf/python/feature_extractor.py

    Args:
      height: The height of feature map.
      width: The width of feature map.
      rf: The receptive field size.
      stride: The effective stride between two adjacent feature points.
      padding: The effective padding size.

    Returns:
      rf_boxes: [N, 4] receptive boxes tensor. Here N equals to height x width.
      Each box is represented by [ymin, xmin, ymax, xmax].
    """

    rf, stride, padding = [196.0, 16.0, 90.0]  # hardcoded for vgg-16 conv5_3

    x, y = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    coordinates = torch.reshape(torch.stack([y, x], dim=2), [-1, 2])
    # [y,x,y,x]
    point_boxes = torch.cat([coordinates, coordinates], 1).type(torch.float32)
    bias = [-padding, -padding, -padding + rf - 1, -padding + rf - 1]
    rf_boxes = stride * point_boxes + torch.FloatTensor(bias)
    return rf_boxes


def calc_keypoint_centers_from_patches(height, width, patch_size, stride):

    H = height // 16  # 16 is the vgg scaling from image space to feature space (conv5)
    W = width // 16
    padding_size = [0, 0]
    patch_size = [patch_size, patch_size]
    stride = [stride, stride]

    Hout = int((H + (2 * padding_size[0]) - patch_size[0]) / stride[0] + 1)
    Wout = int((W + (2 * padding_size[1]) - patch_size[1]) / stride[1] + 1)

    boxes = calc_receptive_boxes(H, W)

    num_regions = Hout * Wout

    k = 0
    keypoints = np.zeros((2, num_regions), dtype=int)
    # Assuming sensible values for stride here, may get errors with large stride values
    for i in range(0, Hout, stride[0]):
        for j in range(0, Wout, stride[1]):
            keypoints[0, k] = ((boxes[j + (i * W), 0] + boxes[(j + (patch_size[1] - 1)) + (i * W), 2]) / 2)
            keypoints[1, k] = ((boxes[j + ((i + 1) * W), 1] + boxes[j + ((i + (patch_size[0] - 1)) * W), 3]) / 2)
            k += 1

    return keypoints


def local_matcher(predictions, query_features_template, index_features_template, batch_size, 
                    patch_size=5, stride=1, height=480, width=640):

    keypoints = calc_keypoint_centers_from_patches(height, width, patch_size, stride)

    reordered_preds = []

    matcher = PatchMatcher(patch_size, stride, keypoints)

    cached_qfile_idx = -1
    for q_idx, pred in enumerate(tqdm(predictions, leave=False, desc='Patch compare pred')):
        diffs = torch.zeros(predictions.shape[1])
        
        qfile_idx = q_idx // batch_size
        if (qfile_idx != cached_qfile_idx):
            cached_qfile_idx = qfile_idx
            qfilename = query_features_template.format(qfile_idx)
            qfile = torch.load(qfilename)['feat']

        qfeat = qfile[q_idx%batch_size]#.cuda() # LD
        
        cached_dbfile_idx = -1
        for k, candidate in enumerate(pred):

            dbfile_idx = candidate // batch_size
            if (dbfile_idx != cached_dbfile_idx):
                cached_dbfile_idx = dbfile_idx
                dbfilename = index_features_template.format(dbfile_idx)
                dbfile = torch.load(dbfilename)['feat']
            
            dbfeat = dbfile[candidate%batch_size].transpose(0, 1)#.cuda() # DL
            diffs[k] = matcher.match(qfeat, dbfeat)

        rerank = np.argsort(diffs, kind='mergesort')
        predictions[q_idx] = pred[rerank]

    return predictions


def local_matcher_tokyo(predictions, query_features_template, index_features_template, batch_size, 
                        query_sizes, patch_size=5, stride=1, height=480, width=640):

    keypoints = calc_keypoint_centers_from_patches(height, width, patch_size, stride)

    reordered_preds = []

    matcher = PatchMatcher(patch_size, stride, keypoints)

    for q_idx, pred in enumerate(tqdm(predictions, leave=False, desc='Patch compare pred')):
        diffs = torch.zeros(predictions.shape[1])
        
        qfilename = query_features_template.format(q_idx)
        qfile = torch.load(qfilename)['feat']

        qfeat = qfile[0]#.cuda() # LD
        if query_sizes[q_idx][0] == 480 and query_sizes[q_idx][1] == 640:
            query_keypoints = keypoints
        else:
            query_keypoints = calc_keypoint_centers_from_patches(
                                query_sizes[q_idx][0], query_sizes[q_idx][1], patch_size, stride)
        
        cached_dbfile_idx = -1
        for k, candidate in enumerate(pred):

            dbfile_idx = candidate // batch_size
            if (dbfile_idx != cached_dbfile_idx):
                cached_dbfile_idx = dbfile_idx
                dbfilename = index_features_template.format(dbfile_idx)
                dbfile = torch.load(dbfilename)['feat']
            
            dbfeat = dbfile[candidate%batch_size].transpose(0, 1)#.cuda() # DL
            diffs[k] = matcher.match(qfeat, dbfeat, query_keypoints)

        rerank = np.argsort(diffs, kind='mergesort')
        predictions[q_idx] = pred[rerank]

    return predictions
