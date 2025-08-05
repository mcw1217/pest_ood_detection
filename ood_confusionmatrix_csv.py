#!/usr/bin/env python
import os
import mmengine
import numpy as np
import pandas as pd
import torch
from numpy.linalg import norm, pinv
from scipy.special import logsumexp, softmax
from sklearn.covariance import EmpiricalCovariance
from PIL import Image
import torchvision as tv
from mmpretrain.apis import init_model
from tqdm import tqdm
import glob
import shutil
from pathlib import Path

# ====== 사용자 설정 ======
cfg_path = 'MOODv2/configs/beit-base-p16_224px.py'
checkpoint_path = 'MOODv2/pretrain/beitv2-base.pth'
fc_path = 'pkl/fc.pkl'
id_train_feature_path = 'pkl/train.pkl'
id_val_feature_path = 'pkl/ID.pkl' 
methods = ['ViM', 'Residual']
fpr = 95 #false positive rate
test_id_dirs = ['test/0']       # test ID 이미지가 들어 있는 폴더들
test_ood_dirs = ['test/1']     # test OOD 이미지가 들어 있는 폴더들
output_root = 'test/result' #confusion matrix folders 저장 장소
csv_name='result.csv' #출력될 csv 이름
# =========================
from pathlib import Path

def collect_images_from_dirs(directories, exts={'.jpg', '.jpeg', '.png', '.bmp', '.tif'}):
    all_paths = []
    exts = {ext.lower() for ext in exts}  # 확장자를 모두 소문자로 통일
    for d in directories:
        for root, dirs, files in os.walk(d):
            for file in files:
                # 파일 확장자 소문자로 변환해서 비교
                if os.path.splitext(file)[1].lower() in exts:
                    all_paths.append(os.path.join(root, file))
    return all_paths


def evaluate(method, score_id, score_ood, target_fpr):
    threshold = np.percentile(score_id, 100 - target_fpr)
    return 'ID' if score_ood >= threshold else 'OOD'

def extract_image_feature(model, cfg, img_path):

    if hasattr(cfg.model.backbone, 'img_size'):
        img_size = cfg.model.backbone.img_size
    else:
        img_size = 224

    transform = tv.transforms.Compose([
        tv.transforms.Resize((img_size, img_size)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()
    with torch.no_grad():
        if cfg.model.backbone.type == 'BEiTPretrainViT':
            feat = model.backbone(image, mask=None)[0].mean(1)
        elif cfg.model.backbone.type == 'SwinTransformer':
            feat = model.backbone(image)[0]
            B, C, H, W = feat.shape
            feat = feat.view(B, C, -1).mean(-1)
        else:
            feat = model.backbone(image)[0]
        return feat.cpu().numpy()

def evaluate_one_image(img_path, feature_id_train, feature_id_val, w, b, u, num_cls):
    logit_id_val = feature_id_val @ w.T + b
    logit_id_train = feature_id_train @ w.T + b
    feature_ood = extract_image_feature(model, cfg, img_path)
    logit_ood = feature_ood @ w.T + b
    softmax_ood = softmax(logit_ood, axis=-1)

    result = {'img_path': img_path}

    # ViM
    if 'ViM' in methods:
        if feature_id_val.shape[-1] >= 2048:
            DIM = num_cls
        elif feature_id_val.shape[-1] >= 768:
            DIM = 512
        else:
            DIM = feature_id_val.shape[-1] // 2
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - u)
        eig_vals, eig_vecs = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray((eig_vecs.T[np.argsort(eig_vals * -1)[DIM:]]).T)

        vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
        alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()

        vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
        energy_id_val = logsumexp(logit_id_val, axis=-1)
        score_id = -vlogit_id_val + energy_id_val

        vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
        energy_ood = logsumexp(logit_ood, axis=-1)
        score_ood = -vlogit_ood + energy_ood

        result['ViM'] = evaluate('ViM', score_id, score_ood, fpr)

    # Residual
    if 'Residual' in methods:
        if feature_id_val.shape[-1] >= 2048:
            DIM = 1000
        elif feature_id_val.shape[-1] >= 768:
            DIM = 512
        else:
            DIM = feature_id_val.shape[-1] // 2
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - u)
        eig_vals, eig_vecs = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray((eig_vecs.T[np.argsort(eig_vals * -1)[DIM:]]).T)

        score_id = -norm(np.matmul(feature_id_val - u, NS), axis=-1)
        score_ood = -norm(np.matmul(feature_ood - u, NS), axis=-1)
        result['Residual'] = evaluate('Residual', score_id, score_ood, fpr)

    return result


cfg = mmengine.Config.fromfile(cfg_path)
model = init_model(cfg, checkpoint_path, device='cuda:0').eval()

# Load all static components
feature_id_train = mmengine.load(id_train_feature_path).squeeze()
feature_id_val = mmengine.load(id_val_feature_path).squeeze()
w, b = mmengine.load(fc_path)
u = -np.matmul(pinv(w), b)
num_cls=len(b)


# ✅ 이미지 경로 수집
id_image_paths = collect_images_from_dirs(test_id_dirs)
ood_image_paths = collect_images_from_dirs(test_ood_dirs)

image_entries = [{'img_path': p, 'true_label': 'ID'} for p in id_image_paths]
image_entries += [{'img_path': p, 'true_label': 'OOD'} for p in ood_image_paths]

results = []
for entry in tqdm(image_entries):
    try:
        result = evaluate_one_image(entry['img_path'], feature_id_train, feature_id_val, w, b, u,num_cls)
        result['true_label'] = entry['true_label']
        results.append(result)
    except Exception as e:
        print(f"[ERROR] {entry['img_path']}: {e}")

df = pd.DataFrame(results)
df.to_csv(csv_name, index=False)
print(f"✅ Saved: {csv_name}.csv")
#print(df.head())

def organize_confusion_matrix(df, output_root):
    """
    Reorganize images based on confusion matrix results for multiple methods.
    
    Parameters:
    - df: pandas DataFrame with columns ['img_path', 'true_label', <method_names>]
    - method_names: list of method column names (e.g., ['ViM', 'Residual'])
    - output_root: the root directory where the confusion matrix folders will be saved
    """
    
    method_names = [col for col in df.columns if col not in ['img_path', 'true_label']]
    for method in method_names:
        print(f"Processing method: {method}")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{method}"):
            img_path = row["img_path"]
            true_label = row["true_label"]  # 'ID' or 'OOD'
            pred_label = row[method]        # 'ID' or 'OOD'

            confusion_type = f"{true_label}-{pred_label}"


            # Extract subfolder structure (e.g., cp_id/cp_0.png)
            rel_path = Path(*Path(img_path).parts[-2:])

            # 출력 경로 설정
            dest_path = Path(output_root) / method / confusion_type / rel_path

            # 디렉토리 생성 및 파일 복사
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_path)

organize_confusion_matrix(df, output_root)