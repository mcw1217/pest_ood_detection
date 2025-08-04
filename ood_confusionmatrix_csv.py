import os
import numpy as np
import pandas as pd
import mmengine
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm
from pathlib import Path
import shutil

# ==== 사용자 설정 =====
train_pkl = 'pkl/train.pkl'
test_id_pkl = 'pkl/test_id_ninco.pkl'
test_ood_pkl = 'pkl/test_ood_ninco.pkl'
fc_path = 'pkl/fc.pkl'
test_id_txt = 'data/NINCO/test_id.txt'
test_ood_txt = 'data/NINCO/test_ood.txt'
output_root = 'data/NINCO/confusion_matrix'
methods = ['ViM', 'Residual']
fpr = 95
csv_name ='per_image_results_eoff_v2.csv'
# ======================

def load_txt(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def evaluate(score_id, score_ood, fpr):
    threshold = np.percentile(score_id, 100 - fpr)
    return np.where(score_ood >= threshold, 'ID', 'OOD')

def main():
    print("Loading features...")
    feature_train = mmengine.load(train_pkl).squeeze()
    feature_id = mmengine.load(test_id_pkl).squeeze()
    feature_ood = mmengine.load(test_ood_pkl).squeeze()
    w, b = mmengine.load(fc_path)
    u = -np.matmul(pinv(w), b)
    num_cls=len(b)

    print("Computing ViM and Residual scores (using test ID for threshold)...")
    D = feature_id.shape[-1]
    if D >= 2048:
        DIM = num_cls 
    elif D >= 768:
        DIM=512 
    else:
        DIM=D // 2

    ec = EmpiricalCovariance(assume_centered=True).fit(feature_train - u)
    eig_vals, eig_vecs = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eig_vecs.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    logit_id = feature_id @ w.T + b
    energy_id = logsumexp(logit_id, axis=-1)
    vlogit_id = norm((feature_id - u) @ NS, axis=-1)
    alpha = logit_id.max(axis=-1).mean() / vlogit_id.mean()

    # ✅ threshold 기준 변경: test ID feature 기준으로 score_id 설정
    score_id_vim = -vlogit_id * alpha + energy_id
    score_id_residual = -norm((feature_id - u) @ NS, axis=-1)

    logit_ood = feature_ood @ w.T + b
    energy_ood = logsumexp(logit_ood, axis=-1)
    vlogit_ood = norm((feature_ood - u) @ NS, axis=-1) * alpha

    score_ood_vim_test = -vlogit_ood + energy_ood
    score_ood_residual_test = -norm((feature_ood - u) @ NS, axis=-1)

    pred_id_vim = evaluate(score_id_vim, score_id_vim, fpr)
    pred_ood_vim = evaluate(score_id_vim, score_ood_vim_test, fpr)

    pred_id_res = evaluate(score_id_residual, score_id_residual, fpr)
    pred_ood_res = evaluate(score_id_residual, score_ood_residual_test, fpr)

    id_paths = load_txt(test_id_txt)
    ood_paths = load_txt(test_ood_txt)
    assert len(id_paths) == len(pred_id_vim)
    assert len(ood_paths) == len(pred_ood_vim)

    df_id = pd.DataFrame({
        'img_path': id_paths,
        'true_label': 'ID',
        'ViM': pred_id_vim,
        'Residual': pred_id_res
    })
    df_ood = pd.DataFrame({
        'img_path': ood_paths,
        'true_label': 'OOD',
        'ViM': pred_ood_vim,
        'Residual': pred_ood_res
    })

    df = pd.concat([df_id, df_ood], ignore_index=True)
    df.to_csv(csv_name, index=False)
    print(f"✅ Saved: {csv_name}")
    return df

if __name__ == '__main__':
    df = main()
    
