#!/usr/bin/env python3
"""

Usage example:
python3 ood_new_inference01.py \
  --fc pkl/fc.pkl \
  --ns_json outputs/ood_new_ns_threshold01.json \
  --id_test_feat pkl/my_cms_id03.pkl \
  --ood_test_feats pkl/my_cms_ood_combine03.pkl \
  --report outputs/ood_new_inference01.json


"""
import argparse
import os
import json
import numpy as np
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn import metrics
import mmengine

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fc', required=True)
    p.add_argument('--ns_json', required=True, help='NS+threshold json produced by build_ns_and_thresholds.py')
    p.add_argument('--id_test_feat', required=True, help='ID test features (for evaluation)')
    p.add_argument('--ood_test_feats', nargs='*', default=[], help='OOD test feature files')
    p.add_argument('--report', default='eval_results.json', help='where to save eval summary')
    return p.parse_args()

def compute_metrics(scores_id, scores_ood):
    # auroc
    y = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])#benchmark의 auc함수의 ind_indicator
    s = np.concatenate([scores_id, scores_ood])#benchmark의 auc함수의 conf
    fpr, tpr, _ = metrics.roc_curve(y, s)
    auroc = float(metrics.auc(fpr, tpr))
    # compute fpr at tpr=0.95
    target_tpr = 0.95
    num_id = len(scores_id)# benchmark의 num_fp_at_recall 함수의 num_ind
    recall_num = int(np.floor(target_tpr * num_id))
    thresh = np.sort(scores_id)[-recall_num]
    num_fp = np.sum(scores_ood >= thresh)
    fpr95 = float(num_fp / max(1, len(scores_ood)))#benchmark의 fpr_recall의 fpr
    return {'auroc': auroc, 'fpr_at_tpr95': fpr95, 'threshold_for_tpr95': float(thresh)}

def main():
    args = parse_args()
    if not os.path.exists(args.fc):
        raise FileNotFoundError(args.fc)
    w, b = mmengine.load(args.fc)
    u = -np.matmul(pinv(w), b)

    ns_info = json.load(open(args.ns_json, 'r'))
    id_test = mmengine.load(args.id_test_feat).squeeze()
    print(f'Loaded id_test: {id_test.shape}')

    ood_tests = {}
    for f in args.ood_test_feats:
        name = os.path.splitext(os.path.basename(f))[0]
        ood_tests[name] = mmengine.load(f).squeeze()
        print(f'Loaded ood_test {name}: {ood_tests[name].shape}')

    report = {}
    # For each method present in ns_json, run inference
    for method, info in ns_info['methods'].items():
        print(f'Running inference for: {method}')
        NS = np.array(ns_info['NS'])
        alpha = info.get('alpha', None)

        # compute ID scores
        if method == 'ViM':
            vlogit_id = norm(np.matmul(id_test - u, NS), axis=-1) * (alpha if alpha is not None else 1.0)
            energy_id = logsumexp((id_test @ w.T + b), axis=-1)
            scores_id = -vlogit_id + energy_id
        else:  # Residual
            scores_id = -norm(np.matmul(id_test - u, NS), axis=-1)

        method_res = {}
        # use chosen_threshold if exists, else compute threshold using id_test vs each ood (fallback)
        chosen_threshold = info.get('threshold', None)

        per_ood_results = {}
        for name, ood_feat in ood_tests.items():
            if method == 'ViM':
                vlogit_o = norm(np.matmul(ood_feat - u, NS), axis=-1) * (alpha if alpha is not None else 1.0)
                energy_o = logsumexp((ood_feat @ w.T + b), axis=-1)
                scores_o = -vlogit_o + energy_o
            else:
                scores_o = -norm(np.matmul(ood_feat - u, NS), axis=-1)

            # evaluate
            metrics_dict = compute_metrics(scores_id, scores_o)

            # if chosen_threshold provided, compute empirical fpr using that threshold
            if chosen_threshold is not None:
                chosen = float(chosen_threshold)
                fp = int(np.sum(scores_o >= chosen))
                fpr_chosen = float(fp / max(1, len(scores_o)))
            else:
                chosen = float(metrics_dict['threshold_for_tpr95'])
                fpr_chosen = float(metrics_dict['fpr_at_tpr95'])

            per_ood_results[name] = {
                'auroc': metrics_dict['auroc'],
                'fpr_at_tpr95': metrics_dict['fpr_at_tpr95'],
                'threshold_tpr95': metrics_dict['threshold_for_tpr95'],
                'chosen_threshold': chosen,
                'fpr_at_chosen_threshold': fpr_chosen
            }
            print(f'{method} | {name} | AUROC {metrics_dict["auroc"]:.4f} | fpr@tpr95 {metrics_dict["fpr_at_tpr95"]:.4f} | chosen_thr {chosen:.6f} | fpr_chosen {fpr_chosen:.4f}')

        method_res['per_ood'] = per_ood_results
        report[method] = method_res

    # save report
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Written eval report to {args.report}')

if __name__ == '__main__':
    main()

'''
Loaded id_test: (18128, 768)
Loaded ood_test my_cms_ood_combine03: (14706, 768)

Running inference for: ViM
ViM | my_cms_ood_combine03 | AUROC 0.8717 | fpr@tpr95 0.3586 | chosen_thr -2.701252 | fpr_chosen 0.2440

Running inference for: Residual
Residual | my_cms_ood_combine03 | AUROC 0.8834 | fpr@tpr95 0.3456 | chosen_thr -8.466538 | fpr_chosen 0.2322

Written eval report to outputs/ood_new_inference01.json
'''