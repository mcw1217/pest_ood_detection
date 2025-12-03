import os
import shutil
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# === 사용자 설정 ===
# root 디렉토리의 경로 (절대경로로 입력하거나 동적으로 설정 가능)
SOURCE_DIR = 'yet'  #../cms_crawling/img

# 프로젝트 내부 저장 폴더
ID_DIR = 'id'
OOD_DIR = 'ood'
UNKNOWN_DIR = 'unknown'

# 시작 위치 선택 
START_INDEX = None          # 정수 입력 (예: 10 → 11번째 이미지부터 시작), None이면 무시
START_FILE = "downloaded_image.jpg"  # START_INDEX가 None일 때만 적용됨

# === 디렉토리 생성 ===
os.makedirs(ID_DIR, exist_ok=True)
os.makedirs(OOD_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# === 이미지 파일 목록 불러오기 ===
image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()  # 정렬 (선택 사항)

current_index = 0  # 현재 보여지는 이미지 인덱스

# === Tkinter GUI 설정 ===
root = tk.Tk()
root.title("image labeling tool")

canvas = tk.Canvas(root, width=900, height=900)#800, 600
canvas.pack()

def load_image(index):
    global image_on_canvas
    canvas.delete("all")  # 이전 이미지 제거

    if index < 0 or index >= len(image_files):
        print("범위를 벗어났습니다.")
        return

    image_path = os.path.join(SOURCE_DIR, image_files[index])
    pil_image = Image.open(image_path)
    canvas.config(width=pil_image.width, height=pil_image.height)

    tk_image = ImageTk.PhotoImage(pil_image)
    image_on_canvas = tk_image
    canvas.create_image(0, 0, anchor='nw', image=tk_image)

def label_image(label):
    if current_index < 0 or current_index >= len(image_files):
        return

    filename = image_files[current_index]
    src_path = os.path.join(SOURCE_DIR, filename)

    # === 기존 위치에서 삭제 (id/ 및 ood/ 및 unknown/)
    for folder in [ID_DIR, OOD_DIR, UNKNOWN_DIR]:
        existing_path = os.path.join(folder, filename)
        if os.path.exists(existing_path):
            os.remove(existing_path)
            print(f"기존 {folder}의 {filename} 삭제됨")

    # === 새로운 위치에 복사
    if label == 'id':
        dst_path = os.path.join(ID_DIR, filename)
    elif label == 'ood':
        dst_path = os.path.join(OOD_DIR, filename)
    elif label == 'unknown':   
        dst_path = os.path.join(UNKNOWN_DIR, filename)
    else:
        return

    shutil.copy2(src_path, dst_path)
    print(f"[{label.upper()}] {filename} 새로 복사됨")

def prev_image():
    global current_index
    if current_index > 0:
        current_index -= 1
        load_image(current_index)

def next_image():
    global current_index
    if current_index < len(image_files) - 1:
        current_index += 1
        load_image(current_index)

def on_key(event):
    key = event.keysym.lower()

    if key == 'left':
        prev_image()
    elif key == 'right':
        next_image()
    elif key == 'd':
        label_image('id')
    elif key == 'f':
        label_image('ood')
    elif key == 'space':
        label_image('unknown')

# === 키 바인딩 ===
root.bind("<Left>", on_key)
root.bind("<Right>", on_key)
root.bind("<d>", on_key)
root.bind("<f>", on_key)
root.bind("<space>", on_key)

# === 첫 이미지 로드 ===
if image_files:
    if START_INDEX is not None:
        if 0 <= START_INDEX < len(image_files):
            current_index = START_INDEX
            print(f"START_INDEX 지정됨 → {image_files[current_index]} (인덱스 {current_index})")
        else:
            current_index = 0
            print(f"START_INDEX 범위 초과, 기본값 시작 → {image_files[current_index]} (인덱스 0)")
    elif START_FILE and START_FILE in image_files:
        current_index = image_files.index(START_FILE)
        print(f"START_FILE 지정됨 → {START_FILE} (인덱스 {current_index})")
    else:
        current_index = 0
        print(f"기본값 시작 → {image_files[current_index]} (인덱스 {current_index})")

    load_image(current_index)
else:
    print("이미지가 존재하지 않습니다.")
    root.quit()

root.mainloop()
