import argparse
import os
import os.path as osp
import numpy as np
import h5py
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import datasets
import models
import evals


def get_data(args):
    root = osp.join(args.data_dir, args.dataset)
    dataset = datasets.Pitts(root, scale='250k') if args.dataset == 'pitts' else datasets.Tokyo(root)

    transform_db = datasets.get_transform()
    transform_q  = datasets.get_transform(tokyo=(args.dataset=='tokyo'))

    q_loader = DataLoader(
        datasets.Preprocessor(dataset.q_test, root=dataset.images_dir, transform=transform_q),
                                batch_size=(1 if args.dataset=='tokyo' else args.test_batch_size), 
                                num_workers=args.workers, shuffle=False, pin_memory=True)
    db_loader = DataLoader(
        datasets.Preprocessor(dataset.db_test, root=dataset.images_dir, transform=transform_db),
                                batch_size=args.test_batch_size, 
                                num_workers=args.workers, shuffle=False, pin_memory=True)

    return q_loader, db_loader, dataset

def get_model(args):
    base_model = models.vgg()
    pool_layer = models.PatchNetVLAD(dim=base_model.feature_dim, num_clusters=args.num_clusters,
                                        patch_size=args.patch_size)
    model = models.EmbedPatchNet(base_model, pool_layer)
    
    ckpt = torch.load(args.resume, map_location='cpu')['state_dict']
    new_ckpt = {}
    for k, v in ckpt.items():
        new_ckpt[k[7:]] = v
    model.load_state_dict(new_ckpt)
    del ckpt, new_ckpt
    return model.cuda()
    
def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    args.gpu = 0
    
    cudnn.benchmark = True

    if not osp.isfile(args.resume):
        raise ValueError('NO pretrained models')

    # Create model
    print("====> constructing model")
    model = get_model(args)
    model.eval()

    pca_parameters_path = osp.join(osp.dirname(args.resume), 'pca_params_'+osp.basename(args.resume).split('.')[0]+'.h5')
    if not osp.isfile(pca_parameters_path):
        raise ValueError('Illegal PCA parameters')
    pca = models.PCA(args.features, (not args.nowhiten), pca_parameters_path)
    pca.load(args.gpu)
    
    to_save = osp.join(args.save_dir, args.dataset+'_patch')
    os.makedirs(to_save, exist_ok=True)

    print("====> loading data")
    q_loader, db_loader, dataset = get_data(args)

    def get_patch_features(loader, name):
        with torch.no_grad():
            for i, (x, _) in enumerate(tqdm(loader, leave=False)):
                x = model(x.cuda()).detach() # N(KC)L
                x = pca.infer(x.transpose(1,2).contiguous())
                filename = '{}_{}.tar'.format(name, i)
                torch.save({'feat': x.cpu()}, osp.join(to_save, filename))
    
    print("====> extracting & saving query features")
    get_patch_features(q_loader, 'query')

    print("====> extracting & saving index features")
    get_patch_features(db_loader, 'index')
    
    del pca
    print("[OK] Patch features extracted.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image-based localization testing")
    # data
    parser.add_argument('--test-batch-size', type=int, default=16,
                        help="tuple numbers in a batch")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--num-clusters', type=int, default=64)
    parser.add_argument('--dataset', type=str, choices=['pitts', 'tokyo'])
    # model
    parser.add_argument('--nowhiten', action='store_true')
    parser.add_argument('--features', type=int, default=128, help='feature dimension after PCA')
    parser.add_argument('--patch-size', type=int, default=5)
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--cuda', type=int, default=0)
    # path
    parser.add_argument('--data-dir', type=str, metavar='PATH', default='datasets')
    parser.add_argument('--save-dir', type=str, metavar='PATH', default='features')
    main()
