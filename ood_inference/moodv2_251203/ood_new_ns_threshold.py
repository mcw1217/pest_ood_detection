#!/usr/bin/env python3
"""

Usage example:
python3 ood_new_ns_threshold01.py \
  --fc pkl/fc.pkl \
  --train_all_feat pkl/plant.pkl \
  --save_json outputs/ood_new_ns_threshold01.json


"""
import argparse
import os
import json
import numpy as np
from numpy.linalg import pinv, norm
from scipy.special import logsumexp, softmax
from sklearn.covariance import EmpiricalCovariance
from sklearn import metrics
import mmengine

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fc', required=True, help='classifier weights file (w,b) for logits')
    p.add_argument('--train_all_feat', required=True, help='ID train ALL features (will split 60/40)')
    #p.add_argument('--ood_feats', nargs='*', default=[], help='OOD feature files (for thresholding)')
    p.add_argument('--save_json', required=True, help='where to write NS+threshold json')
    p.add_argument('--split_seed', type=int, default=420, help='random seed for 60/40 split')
    return p.parse_args()
'''
def fpr_recall(ind_conf, ood_conf, tpr=0.95):
    num_ind = len(ind_conf)
    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1
    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]#id/ood 구분 threshold(id 중 상위 95%)
    num_fp = np.sum(ood_conf >= thresh)
    fpr = num_fp / max(1, len(ood_conf))
    return fpr, thresh
'''
'''
def auc_score(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    labels = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))
    fpr, tpr, _ = metrics.roc_curve(labels, conf)
    return float(metrics.auc(fpr, tpr))
'''

def main():
    args = parse_args()
    # load fc weights if provided
    if not os.path.exists(args.fc):
        raise FileNotFoundError(args.fc)
    w, b = mmengine.load(args.fc)
    print(f'{w.shape=}, {b.shape=}')#w.shape=(1000, 768), b.shape=(1000,)
    num_cls = len(b)

    # load features
    feats_all = mmengine.load(args.train_all_feat).squeeze()
    print(f'Loaded train_all features: {feats_all.shape}')#Loaded train_all features: (285957, 768)

    '''
    # OOD feature dict
    ood_dict = {}
    for f in args.ood_feats:
        name = os.path.splitext(os.path.basename(f))[0]
        ood_dict[name] = mmengine.load(f).squeeze()
        print(f'Loaded OOD {name}: {ood_dict[name].shape}')
    '''

    # compute u = -pinv(w) b (same as ViM)
    u = -np.matmul(pinv(w), b)

    # split train_all -> train6(60%) and val4(40%)
    rng = np.random.RandomState(args.split_seed)
    n_all = len(feats_all)
    perm = rng.permutation(n_all)
    split_idx = int(np.floor(0.6 * n_all))
    if split_idx < 1 or split_idx >= n_all:
        raise ValueError("Not enough samples to split; check train_all size.")
    feat_train6 = feats_all[perm[:split_idx]]
    feat_val4 = feats_all[perm[split_idx:]]
    print(f'train6: {feat_train6.shape}, val4: {feat_val4.shape}')#train6: (171574, 768), val4: (114383, 768)

    #results = {}
    results = {'NS': None, 'methods': {}}

    # method: ViM
    for method in ['ViM', 'Residual']:
        print(f'--- Building for method {method}')
        # determine DIM (principal-class dim) heuristics from original code style
        feat_dim = feat_val4.shape[-1]
        print(f'feat_dim of val4: {feat_val4.shape[-1]}')#둘 다 feat_dim of val4: 768
        if feat_dim >= 2048:
            DIM = num_cls if method == 'ViM' else 1000
        elif feat_dim >= 768:
            DIM = 512
        else:
            DIM = feat_dim // 2

        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feat_train6 - u)
        eig_vals, eig_vecs = np.linalg.eig(ec.covariance_)
        # noise subspace = eigenvectors corresponding to smaller eigenvalues
        NS = np.ascontiguousarray((eig_vecs.T[np.argsort(eig_vals * -1)[DIM:]]).T)
        results['NS'] = NS.tolist()

        # compute alpha for ViM
        alpha = None
        if method == 'ViM':
            logits_train6 = feat_train6 @ w.T + b
            vlogit_train6 = norm(np.matmul(feat_train6 - u, NS), axis=-1)
            alpha = logits_train6.max(axis=-1).mean() / vlogit_train6.mean()
        # compute scores on val4 (ID split)
        if method == 'ViM':
            vlogit_val4 = norm(np.matmul(feat_val4 - u, NS), axis=-1) * alpha
            energy_val4 = logsumexp((feat_val4 @ w.T + b), axis=-1)#logits_val4가 안에 계산돼 있음
            score_id_val4 = -vlogit_val4 + energy_val4
            
        else:  # Residual
            score_id_val4 = -norm(np.matmul(feat_val4 - u, NS), axis=-1)
        
        recall_num = int(np.floor(0.95 * len(score_id_val4)))#0.95=tpr
        thresh = np.sort(score_id_val4)[-recall_num]#id/ood 구분 threshold(id 중 상위 95%)
        print(f'{method}: shape of score_id_val4={score_id_val4.shape}, tpr95 thresh={thresh:.6f}')
        #ViM: shape of score_id_val4=(114383,), tpr95 thresh=-2.701252
        #Residual: shape of score_id_val4=(114383,), tpr95 thresh=-8.466538
        '''
        # compute thresholds per OOD
        thresholds_per_ood = {}
        for name, feato in ood_dict.items():
            if method == 'ViM':
                vlogit_o = norm(np.matmul(feato - u, NS), axis=-1) * alpha
                energy_o = logsumexp((feato @ w.T + b), axis=-1)
                score_o = -vlogit_o + energy_o
            else:
                score_o = -norm(np.matmul(feato - u, NS), axis=-1)

            #fpr_o, thresh = fpr_recall(score_id_val4, score_o, tpr=0.95)
            recall_num = int(np.floor(0.95 * len(score_id_val4)))#0.95=tpr
            thresh = np.sort(score_id_val4)[-recall_num]#id/ood 구분 threshold(id 중 상위 95%)
            thresholds_per_ood[name] = thresh
            print(f'{method} - {name}: fpr@tpr95={fpr_o:.4f}, thresh={thresh:.6f}')

        # choose aggregated threshold: median across OOD thresholds (user can change to mean/max)
        if len(thresholds_per_ood) > 0:
            chosen = float(np.median(list(thresholds_per_ood.values())))
        else:
            # fallback if no ood provided: use median of id val scores
            chosen = float(np.median(score_id_val4))
        '''
        results['methods'][method] = {
            'alpha': None if alpha is None else float(alpha),
            'threshold': float(thresh)
        }
        #print(f'{method} chosen_threshold = {chosen:.6f}')
        
    '''
    # NS와 threshold 값들을 float로 변환하는 함수
    def convert_to_float(obj):
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)  # numpy 타입을 float로 변환
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # numpy 배열을 리스트로 변환
        elif isinstance(obj, dict):
            return {key: convert_to_float(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_float(item) for item in obj]
        return obj
    # 변환된 results를 JSON으로 저장
    with open(output_json, 'w') as f:
        json.dump(convert_to_float(results), f, indent=2)

    print(f'JSON file saved: {output_json}')
    '''

    # save JSON
    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    with open(args.save_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Written NS+thresholds JSON to: {args.save_json}')

if __name__ == '__main__':
    main()
