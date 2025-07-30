# ood_folder_split.py
import argparse
import os
import shutil
import numpy as np
import mmengine
from numpy.linalg import norm, pinv
from scipy.special import logsumexp, softmax
from sklearn.covariance import EmpiricalCovariance

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--fc', required=True)
    parser.add_argument('--id_train_feature', required=True)
    parser.add_argument('--id_val_feature', required=True)
    parser.add_argument('--ood_feature', required=True)
    parser.add_argument('--train_label', required=True)
    parser.add_argument('--fpr', type=int, default=95)
    parser.add_argument('--methods', nargs='+', default=['ViM', 'Residual'])
    parser.add_argument('--img_roots', nargs='+', required=True)
    parser.add_argument('--output_dir', default='results')
    return parser.parse_args()

def evaluate(method, score_id, score_ood, target_fpr):
    threshold = np.percentile(score_id, 100 - target_fpr)
    return 'in-distribution' if score_ood >= threshold else 'out-of-distribution'

def prepare_ood_detection(args):
    feature_id_train = mmengine.load(args.id_train_feature).squeeze()
    feature_id_val = mmengine.load(args.id_val_feature).squeeze()
    feature_ood = mmengine.load(args.ood_feature)
    w, b = mmengine.load(args.fc)
    num_cls = len(b)

    try:
        train_label_lines = mmengine.list_from_file(args.train_label)
        train_labels = np.array([int(line.rsplit(' ', 1)[-1]) for line in train_label_lines], dtype=int)
    except ValueError:
        print("Detected label-less train_label file. All labels set to 0.")
        train_labels = np.zeros(len(train_label_lines), dtype=int)

    u = -np.matmul(pinv(w), b)

    score_id_dict = {}
    common_data = {}

    if 'ViM' in args.methods:
        DIM = 512 if feature_id_val.shape[-1] >= 768 else feature_id_val.shape[-1] // 2
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
        vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
        alpha = (feature_id_train @ w.T + b).max(axis=-1).mean() / vlogit_id_train.mean()
        vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
        energy_id_val = logsumexp(feature_id_val @ w.T + b, axis=-1)
        score_id = -vlogit_id_val + energy_id_val
        score_id_dict['ViM'] = score_id
        common_data['vim'] = dict(NS=NS, u=u, alpha=alpha, w=w, b=b)

    if 'Residual' in args.methods:
        DIM = 512 if feature_id_val.shape[-1] >= 768 else feature_id_val.shape[-1] // 2
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
        score_id = -norm(np.matmul(feature_id_val - u, NS), axis=-1)
        score_id_dict['Residual'] = score_id
        common_data['residual'] = dict(NS=NS, u=u)

    return feature_id_val, feature_ood, score_id_dict, common_data

def main():
    args = parse_args()
    feature_id_val, feature_ood_all, score_id_dict, common_data = prepare_ood_detection(args)

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    id_idx = 0  # ID 피처 인덱스, 필요하면 만듭니다 (ID val feature 수에 맞춰)
    ood_idx = 0  # 기존 인덱스

    for i, root in enumerate(args.img_roots):
        actual = 'ID' if i == 0 else 'OOD'
        for dirpath, _, files in os.walk(root):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() not in img_exts:
                    continue

                if actual == 'ID':
                    # ID 이미지 feature 인덱스 사용 (feature_id_val 기준)
                    feature = feature_id_val[id_idx].reshape(1, -1)
                    id_idx += 1
                else:
                    feature = feature_ood_all[ood_idx].reshape(1, -1)
                    ood_idx += 1

                for method in args.methods:
                    score_id = score_id_dict[method]

                    if method == 'ViM':
                        data = common_data['vim']
                        energy = logsumexp(feature @ data['w'].T + data['b'], axis=-1)
                        vlogit = norm(np.matmul(feature - data['u'], data['NS']), axis=-1) * data['alpha']
                        score_ood = -vlogit + energy
                    elif method == 'Residual':
                        data = common_data['residual']
                        score_ood = -norm(np.matmul(feature - data['u'], data['NS']), axis=-1)

                    label = evaluate(method, score_id, score_ood, args.fpr)
                    rel = os.path.relpath(dirpath, root)
                    dst_dir = os.path.join(args.output_dir, method, f"{actual}-{label}", rel)
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy(os.path.join(dirpath, fn), os.path.join(dst_dir, fn))
                    print(f"[{method}] {os.path.join(dirpath, fn)} → {dst_dir}/{fn}")


                for method in args.methods:
                    score_id = score_id_dict[method]

                    if method == 'ViM':
                        data = common_data['vim']
                        energy = logsumexp(feature @ data['w'].T + data['b'], axis=-1)
                        vlogit = norm(np.matmul(feature - data['u'], data['NS']), axis=-1) * data['alpha']
                        score_ood = -vlogit + energy
                    elif method == 'Residual':
                        data = common_data['residual']
                        score_ood = -norm(np.matmul(feature - data['u'], data['NS']), axis=-1)

                    label = evaluate(method, score_id, score_ood, args.fpr)
                    rel = os.path.relpath(dirpath, root)
                    dst_dir = os.path.join(args.output_dir, method, f"{actual}-{label}", rel)
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy(os.path.join(dirpath, fn), os.path.join(dst_dir, fn))
                    print(f"[{method}] {os.path.join(dirpath, fn)} → {dst_dir}/{fn}")

if __name__ == '__main__':
    main()
