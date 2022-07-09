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
    
    pred_dir = args.pred_dir
    feat_dir = osp.join(args.feat_dir, args.dataset+'_patch')

    print('====> loading predictions')
    filename = '{}_pred_regional.tar'.format(args.dataset)
    ranks = torch.load(osp.join(pred_dir, filename))['pred']
    
    print("====> reranking")
    query_features_template = 'query_{}.tar'
    index_features_template = 'index_{}.tar'

    if args.dataset == 'tokyo':
        '''
        For Tokyo, the queries are of different sizes.
        Thus, we explicitly calculate the image size of each query.
        The sizes are saved as an array of shape N*2, 
            where N is the number of queries and the size is of format (height, width).

        '''
        qsize = torch.load('./files/tokyo_query_size.tar')['size']
        reranks = evals.local_matcher_tokyo(ranks[:, :args.topk], 
                            osp.join(feat_dir, query_features_template), 
                            osp.join(feat_dir, index_features_template),
                            args.test_batch_size, qsize)
    else:
        reranks = evals.local_matcher(ranks[:, :args.topk], 
                            osp.join(feat_dir, query_features_template), 
                            osp.join(feat_dir, index_features_template),
                            args.test_batch_size)

    def calc_recalls(rank, desc=''):
        # evaluate
        print("====> evaluating " + desc)
        
        recall_topks = [1, 5, 10]
        metrics = evals.evaluate(rank, dataset.test_pos, dataset.db_test, 
                    recall_topk=recall_topks, nms=(args.dataset=='tokyo'))
        for k in recall_topks:
            line = 'Recall@{}: {:.3f}'.format(k, metrics['recall@{}'.format(k)])
            print(line)

    ranks[:, :args.topk] = reranks
    calc_recalls(ranks, 'spatial verification')

    print("====> saving predictions")
    filename = '{}_pred_local.tar'.format(args.dataset)
    torch.save({'pred': ranks}, osp.join(pred_dir, filename))


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
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--patch-size', type=int, default=5)
    # path
    # path
    parser.add_argument('--data-dir', type=str, metavar='PATH', default='datasets')
    parser.add_argument('--feat-dir', type=str, metavar='PATH', default='features')
    parser.add_argument('--pred-dir', type=str, metavar='PATH', default='predictions')
    main()
