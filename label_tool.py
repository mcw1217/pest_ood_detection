import os
import shutil
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# === 사용자 설정 ===
# root/file 디렉토리의 경로 (절대경로로 입력하거나 동적으로 설정 가능)
SOURCE_DIR = '../cms_crawling/img'  # 여기를 실제 경로로 바꿔줘

# 프로젝트 내부 저장 폴더
ID_DIR = 'id_new'
OOD_DIR = 'ood_new'

# === 디렉토리 생성 ===
os.makedirs(ID_DIR, exist_ok=True)
os.makedirs(OOD_DIR, exist_ok=True)

# === 이미지 파일 목록 불러오기 ===
image_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()  # 정렬 (선택 사항)

current_index = 0  # 현재 보여지는 이미지 인덱스

# === Tkinter GUI 설정 ===
root = tk.Tk()
root.title("image labeling tool")

canvas = tk.Canvas(root, width=800, height=600)
canvas.pack()

#image_on_canvas = None

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

    # === 기존 위치에서 삭제 (id/ 및 ood/)
    for folder in [ID_DIR, OOD_DIR]:
        existing_path = os.path.join(folder, filename)
        if os.path.exists(existing_path):
            os.remove(existing_path)
            print(f"기존 {folder}의 {filename} 삭제됨")

    # === 새로운 위치에 복사
    if label == 'id':
        dst_path = os.path.join(ID_DIR, filename)
    elif label == 'ood':
        dst_path = os.path.join(OOD_DIR, filename)
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

# === 키 바인딩 ===
root.bind("<Left>", on_key)
root.bind("<Right>", on_key)
root.bind("<d>", on_key)
root.bind("<f>", on_key)

# === 첫 이미지 로드 ===
if image_files:
    load_image(current_index)
else:
    print("이미지가 존재하지 않습니다.")
    root.quit()

root.mainloop()
