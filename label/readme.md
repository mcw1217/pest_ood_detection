## ood_confusionmatrix_csv.py
ood_confusionmatrix_csv.py 상단    
'사용자 설정'에서   
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
부분에 입력 후 실행

## label_tool.py
START_INDEX: 시작하고 싶은 사진 index (None이면 처음부터)
START_FILE: 시작하고 싶은 파일명 (START_INDEX가 None일 때만 적용됨)
이미지 하나가 뜨면   
D 누르면 id/ 폴더에 복사   
F 누르면 ood/ 폴더에 복사
스페이스바 누르면 unknown/ 폴더에 복사(모호한 것들)  
← 누르면 이전 이미지로 감    
→ 누르면 다음 이미지로 감    
(방향키는 복사 안 함)   
이전에 라벨링했던 이미지에서 D나 F를 누르면   
예전에 라벨링되어 있던 것이 삭제되고 새로 라벨링한 폴더에 다시 복사됨
(기존 라벨을 무시하고 새로운 라벨로 덮어쓰기)

## ood_label_folder.py
label_tool.py에서 ood 폴더 내 이미지들을
라벨이 폴더명인 폴더들로 이동(오분류했을 경우를 위해 id, unknown 옵션도 있음)
화면에 뜨는 버튼 클릭해서 라벨링