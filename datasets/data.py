import os.path as osp
import numpy as np
import copy
from sklearn.neighbors import NearestNeighbors
import json


__all__ = ['Pitts', 'Tokyo']


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def _pluck(identities, utm, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        x, y = utm[pid]
        for fname in pid_images:
            if relabel:
                ret.append((fname, index, x, y))
            else:
                ret.append((fname, pid, x, y))
    
    return sorted(ret)


def get_groundtruth(query, gallery, intra_thres):
    utm_query = [[u[2], u[3]] for u in query]
    utm_gallery = [[u[2], u[3]] for u in gallery]
    neigh = NearestNeighbors(n_jobs=-1)
    neigh.fit(utm_gallery)
    dist, neighbors = neigh.radius_neighbors(utm_query, radius=intra_thres)
    
    pos, select_pos = [], []
    for idx, p in enumerate(neighbors):
        pid = query[idx][1]
        select_p = [i for i in p.tolist() if gallery[i][1]!=pid]
        if (len(select_p)>0):
            pos.append(select_p)
            select_pos.append(idx)

    return pos, select_pos


class BaseDataset(object):
    def __init__(self, root, intra_thres=25):
        self.root = root
        self.intra_thres = intra_thres
        self.q_test, self.db_test = [], []

    @property
    def images_dir(self):
        return osp.join(self.root, 'raw')

    def load(self, verbose, scale=None):
        if (scale is None):
            splits = read_json(osp.join(self.root, 'splits.json'))
            meta = read_json(osp.join(self.root, 'meta.json'))
        else:
            splits = read_json(osp.join(self.root, 'splits_'+scale+'.json'))
            meta = read_json(osp.join(self.root, 'meta_'+scale+'.json'))
        identities = meta['identities']
        utm = meta['utm']

        q_test_pids = sorted(splits['q_test'])
        db_test_pids = sorted(splits['db_test'])

        self.q_test = _pluck(identities, utm, q_test_pids, relabel=False)
        self.db_test = _pluck(identities, utm, db_test_pids, relabel=False)

        self.test_pos, select = get_groundtruth(self.q_test, self.db_test, self.intra_thres)
        assert(len(select)==len(self.q_test))

        if (verbose):
            print(self.__class__.__name__, "test dataset loaded")
            print("  subset        | # pids | # images")
            print("  ---------------------------------")
            print("  test_query    | {:5d}  | {:8d}"
                  .format(len(q_test_pids), len(self.q_test)))
            print("  test_gallery  | {:5d}  | {:8d}"
                  .format(len(db_test_pids), len(self.db_test)))

    def _check_integrity(self, scale=None):
        if (scale is None):
            return osp.isfile(osp.join(self.root, 'meta.json')) and \
                   osp.isfile(osp.join(self.root, 'splits.json'))
        else:
            return osp.isfile(osp.join(self.root, 'meta_'+scale+'.json')) and \
                   osp.isfile(osp.join(self.root, 'splits_'+scale+'.json'))


class Pitts(BaseDataset):

    def __init__(self, root, scale='250k', verbose=True, **kwargs):
        super(Pitts, self).__init__(root, **kwargs)
        self.scale = scale

        self.arrange()
        self.load(verbose, scale)

    def arrange(self):
        assert self._check_integrity(self.scale)


class Tokyo(BaseDataset):

    def __init__(self, root, scale=None, verbose=True, **kwargs):
        super(Tokyo, self).__init__(root, **kwargs)

        self.arrange()
        self.load(verbose)

    def arrange(self):
        assert self._check_integrity()
