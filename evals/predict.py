import torch
import torch.nn.functional as F 
import numpy as np
from tqdm.auto import tqdm


__all__ = ['extract_features', 'rank_embeddings_region']


def extract_features(model, loader, gpu, vecdim, regiondim=0, pca=None):
    features = torch.zeros(len(loader.sampler), regiondim, vecdim)
    # prepare for pca
    if (pca is not None):
        pca.load(gpu=gpu)
    # run model
    model.eval()
    bs = loader.batch_size
    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(loader, leave=False)):
            x = F.normalize(model(x.cuda(gpu)).detach(), p=2, dim=-1)
            if (pca is not None):
                x = pca.infer(x)
            features[i*bs:(i+1)*bs] = x.cpu()
    # free pca
    if (pca is not None):
        del pca
    return features


def rank_embeddings_region(qvecs, dbvecs, topk=100, query_region=True):
    if topk < 1:
        raise ValueError('Illegal topk {} to rerank'.format(topk))

    sub_qvecs = qvecs[:, -1, :] # full map
    sub_dbvecs = dbvecs[:, -1, :] # full map
    dists = torch.mm(sub_qvecs, sub_dbvecs.t())
    ranks = torch.argsort(dists, dim=1, descending=True)
    # return ranks
    
    sub_ranks = ranks[:, :topk].clone()
    dists = dists.gather(1, sub_ranks)
    # check the query-to-region similarity, use the largest similarity
    # m is the number of regions
    for i in range(qvecs.size(0)):
        x = qvecs[i, :] if query_region else sub_qvecs[i, :].unsqueeze(0) # m*D
        y = dbvecs[sub_ranks[i], :, :].view(-1, x.size(-1)) # k*m*D ==> mk*D
        tmp = torch.mm(y, x.t()) # mk*m
        tmp, _ = torch.max(tmp.view(topk, -1), dim=1) # mk*m ==> k*mm
        dists[i] = torch.max(tmp, dists[i])
   
    reranks = torch.argsort(dists, dim=1, descending=True)
    sub_ranks = sub_ranks.gather(1, reranks)

    ranks = ranks.cpu()
    ranks_region = ranks.clone()
    ranks_region[:, :topk] = sub_ranks.cpu()

    return ranks, ranks_region
