import numpy as np


__all__ = ['evaluate']


def spatial_nms(pred, db_ids, topN):
    pred_select = pred[:topN]
    pred_pids = [db_ids[i] for i in pred_select]
    # find unique
    seen = set()
    seen_add = seen.add
    pred_pids_unique = [i for i, x in enumerate(pred_pids) if not (x in seen or seen_add(x))]
    return [pred_select[i] for i in pred_pids_unique]


def evaluate(sort_idx, gt, gallery, recall_topk=[1, 5, 10], nms=False):
    db_ids = [db[1] for db in gallery]

    correct_at_n = np.zeros(len(recall_topk))

    for qIx, pred in enumerate(sort_idx):
        if (nms):
            pred = spatial_nms(pred.tolist(), db_ids, max(recall_topk)*12)

        for i, n in enumerate(recall_topk):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    
    recalls = correct_at_n / len(gt)
    del sort_idx

    metrics = {}
    for i, k in enumerate(recall_topk):
        metrics['recall@{}'.format(k)] = recalls[i]

    return metrics
