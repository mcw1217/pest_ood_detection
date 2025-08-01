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
test_id_pkl = 'pkl/test_id_ninco.pkl'#'pkl/test_id_ninco.pkl'
test_ood_pkl = 'pkl/test_ood_ninco.pkl'#'pkl/test_ood_ninco.pkl'
fc_path = 'pkl/fc.pkl'
test_id_txt = 'data/NINCO/test_id.txt'#'data/NINCO/test_id.txt'
test_ood_txt = 'data/NINCO/test_ood.txt'#'data/NINCO/test_ood.txt'
output_root = 'data/NINCO/pkl01'#다른 형태의 pkl이면 pkl_new01
methods = ['ViM', 'Residual']
fpr = 95
csv_name='evaluate_ood_pkl01.csv'
# ======================


def load_txt(txt_path):
    base_dir = os.path.splitext(txt_path)[0]
    with open(txt_path, 'r') as f:
        return [os.path.join(base_dir, line.strip()) for line in f.readlines()]
        '''

def load_txt(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f.readlines()]
'''
def evaluate(score_train, score_input, fpr):
    threshold = np.percentile(score_train, 100 - fpr)
    return np.where(score_input >= threshold, 'ID', 'OOD')

def main():
    # 1. Load all features
    feature_train = mmengine.load(train_pkl).squeeze()
    feature_id = mmengine.load(test_id_pkl).squeeze()
    feature_ood = mmengine.load(test_ood_pkl).squeeze()
    w, b = mmengine.load(fc_path)
    u = -np.matmul(pinv(w), b)

    # 2. Compute ViM/Residual shared params
    D = feature_train.shape[-1]
    DIM = 1000 if D >= 2048 else 512 if D >= 768 else D // 2

    ec = EmpiricalCovariance(assume_centered=True).fit(feature_train - u)
    eig_vals, eig_vecs = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eig_vecs.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    # ✅ threshold 기준을 test_id 기준으로 바꿈
    logit_id = feature_id @ w.T + b
    energy_id = logsumexp(logit_id, axis=-1)
    vlogit_id = norm((feature_id - u) @ NS, axis=-1)
    alpha = logit_id.max(axis=-1).mean() / vlogit_id.mean()

    # ✅ 기준 점수: test_id
    score_th_vim = -vlogit_id * alpha + energy_id
    score_th_residual = -norm((feature_id - u) @ NS, axis=-1)

    # 3. Compute scores for test sets
    logit_ood = feature_ood @ w.T + b
    energy_ood = logsumexp(logit_ood, axis=-1)
    vlogit_ood = norm((feature_ood - u) @ NS, axis=-1) * alpha
    score_ood_vim = -vlogit_ood + energy_ood
    score_ood_residual = -norm((feature_ood - u) @ NS, axis=-1)

    # 4. Apply threshold
    pred_id_vim = evaluate(score_th_vim, score_th_vim, fpr)
    pred_ood_vim = evaluate(score_th_vim, score_ood_vim, fpr)

    pred_id_res = evaluate(score_th_residual, score_th_residual, fpr)
    pred_ood_res = evaluate(score_th_residual, score_ood_residual, fpr)

    # 5. Build result dataframe
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
    print("✅ Saved: per_image_results.csv")
    return df

def organize_confusion_matrix(df, output_root):
    method_names = [col for col in df.columns if col not in ['img_path', 'true_label']]
    for method in method_names:
        print(f"Organizing: {method}")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=method):
            img_path = row['img_path']
            true_label = row['true_label']
            pred_label = row[method]
            confusion_type = f"{true_label}-{pred_label}"
            rel_path = Path(*Path(img_path).parts[-2:])
            dest_path = Path(output_root) / method / confusion_type / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(img_path, dest_path)
            except Exception as e:
                print(f"[ERROR] Could not copy {img_path}: {e}")

if __name__ == '__main__':
    df = main()
    organize_confusion_matrix(df, output_root)
