import argparse
import os
import os.path as osp
import numpy as np

import torch

import datasets
import models
import evals


def get_data(args):
    root = osp.join(args.data_dir, args.dataset)
    dataset = datasets.Pitts(root, scale='250k') if args.dataset == 'pitts' else datasets.Tokyo(root)

    return dataset

    
def main():
    args = parser.parse_args()

    dataset = get_data(args)
    
    print("====> loading query features")
    filename = '{}_query.tar'.format(args.dataset)
    q_vecs = torch.load(osp.join(args.load_dir, filename))['feat']

    print("====> loading index features")
    filename = '{}_index.tar'.format(args.dataset)
    db_vecs = torch.load(osp.join(args.load_dir, filename))['feat']

    def calc_recalls(rank, desc=''):
        # evaluate
        print("====> evaluating " + desc)
        
        recall_topks = [1, 5, 10]
        metrics = evals.evaluate(rank, dataset.test_pos, dataset.db_test, 
                    recall_topk=recall_topks, nms=(args.dataset=='tokyo'))
        for k in recall_topks:
            line = 'Recall@{}: {:.3f}'.format(k, metrics['recall@{}'.format(k)])
            print(line)
    
    print("====> predicting")
    ranks, ranks_region = evals.rank_embeddings_region(q_vecs, db_vecs, topk=args.topk, query_region=(args.dataset!='pitts'))
    del q_vecs, db_vecs

    ranks, ranks_region = ranks[:, :120].contiguous(), ranks_region[:, :120].contiguous()
    calc_recalls(ranks, 'baseline (global retrieval)')
    calc_recalls(ranks_region, 'region refinement')

    print("====> saving predictions")
    os.makedirs(args.save_dir, exist_ok=True)
    filename = '{}_pred_base.tar'.format(args.dataset)
    torch.save({'pred': ranks}, osp.join(args.save_dir, filename))
    filename = '{}_pred_regional.tar'.format(args.dataset)
    torch.save({'pred': ranks_region}, osp.join(args.save_dir, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image-based localization testing")
    # data
    parser.add_argument('--test-batch-size', type=int, default=32,
                        help="tuple numbers in a batch")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--num-clusters', type=int, default=64)
    parser.add_argument('--dataset', type=str, choices=['pitts', 'tokyo'])
    # model
    parser.add_argument('--topk', type=int, default=100)
    # path
    parser.add_argument('--data-dir', type=str, metavar='PATH', default='datasets')
    parser.add_argument('--load-dir', type=str, metavar='PATH', default='features')
    parser.add_argument('--save-dir', type=str, metavar='PATH', default='predictions')
    main()
