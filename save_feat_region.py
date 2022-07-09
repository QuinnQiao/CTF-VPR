import argparse
import os
import os.path as osp
import numpy as np
import h5py

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
    pool_layer = models.NetVLAD(dim=base_model.feature_dim, num_clusters=args.num_clusters)
    model = models.EmbedRegionNet(base_model, pool_layer)
    
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

    pca_parameters_path = osp.join(osp.dirname(args.resume), 'pca_params_'+osp.basename(args.resume).split('.')[0]+'.h5')
    if not osp.isfile(pca_parameters_path):
        raise ValueError('Illegal PCA parameters')
    pca = models.PCA(args.features, (not args.nowhiten), pca_parameters_path)
    
    os.makedirs(args.save_dir, exist_ok=True)

    print("====> loading data")
    q_loader, db_loader, dataset = get_data(args)
    
    print("====> extracting query features")
    model.reset_ratio(args.query_subratio)
    q_vecs = evals.extract_features(model, q_loader, args.gpu, args.features, regiondim=7, pca=pca)
    filename = '{}_query.tar'.format(args.dataset)
    print("====> saving query features")
    torch.save({'feat': q_vecs}, osp.join(args.save_dir, filename))
    del q_vecs

    print("====> extracting index features")
    model.reset_ratio(args.index_subratio)
    db_vecs = evals.extract_features(model, db_loader, args.gpu, args.features, regiondim=7, pca=pca)
    filename = '{}_index.tar'.format(args.dataset)
    print("====> saving index features")
    torch.save({'feat': db_vecs}, osp.join(args.save_dir, filename))
    del db_vecs

    print("[OK] Global & regional features extracted.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image-based localization testing")
    # data
    parser.add_argument('--test-batch-size', type=int, default=32,
                        help="tuple numbers in a batch")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--num-clusters', type=int, default=64)
    parser.add_argument('--dataset', type=str, choices=['pitts', 'tokyo'])
    # model
    parser.add_argument('--query-subratio', type=float, default=0.75)
    parser.add_argument('--index-subratio', type=float, default=0.65)
    parser.add_argument('--nowhiten', action='store_true')
    parser.add_argument('--features', type=int, default=4096, help='feature dimension after PCA')
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--cuda', type=int, default=0)
    # path
    parser.add_argument('--data-dir', type=str, metavar='PATH', default='datasets')
    parser.add_argument('--save-dir', type=str, metavar='PATH', default='features')
    main()
