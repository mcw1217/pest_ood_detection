import os
import shutil
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# === 사용자 설정 ===
ROOT_DIR = "./ood"   # root 폴더 경로
LABELS = {
    "IN-ON-A-0": "물체 x, 색상",
    "IN-ON-B-0": "물체 x, 패턴",
    "IN-OY-X-0": "물체 O, 농작물 x",
    "IY-RN-A-0": "인터넷 사진",#(워터마크)
    "IY-RN-B-0": "그림",
    "IY-RN-C-0": "인쇄물",
    "IY-RY-A-0": "중앙객체 X",#(농작물이 배경에)
    "IY-RY-B-0": "중앙객체 작음",#안 보여(
    "IY-RY-C-0": "다른 색",#다른 색의 영향 받음
    "IY-RY-D-1": "흔들림",#흔들린 농작물(블러 처리 수준)
    "IY-RY-D-2": "초점 안 맞음",
    "IY-RY-D-3": "가려짐/알 수 없음",#가려지거나 알 수 없음
    "id": "id",
    "unknown": "unknown"
}

# === 라벨 폴더 생성 ===
for folder in LABELS.keys():
    os.makedirs(os.path.join(ROOT_DIR, folder), exist_ok=True)

# === 이미지 파일 목록 불러오기 (root 바로 아래) ===
image_files = [f for f in os.listdir(ROOT_DIR) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))
               and os.path.isfile(os.path.join(ROOT_DIR, f))]
image_files.sort()

current_index = 0

# === Tkinter GUI 설정 ===
root = tk.Tk()
root.title("button Image Labeling Tool")

# 좌측: 이미지 영역
canvas = tk.Canvas(root)
canvas.grid(row=0, column=0, rowspan=len(LABELS), padx=10, pady=10)

# 전역 변수
image_on_canvas = None


def load_image(index):
    """이미지를 로드해서 캔버스에 표시"""
    global image_on_canvas, current_index
    canvas.delete("all")

    if index < 0 or index >= len(image_files):
        messagebox.showinfo("완료", "모든 이미지를 확인했습니다.")
        return

    current_index = index
    image_path = os.path.join(ROOT_DIR, image_files[index])
    pil_image = Image.open(image_path)

    # 캔버스 크기를 이미지 크기에 맞게 조정
    canvas.config(width=pil_image.width, height=pil_image.height)

    tk_image = ImageTk.PhotoImage(pil_image)
    image_on_canvas = tk_image
    canvas.create_image(0, 0, anchor="nw", image=tk_image)


def move_image(label):
    """현재 이미지를 라벨 폴더로 이동"""
    global current_index
    if current_index < 0 or current_index >= len(image_files):
        return

    filename = image_files[current_index]
    src_path = os.path.join(ROOT_DIR, filename)
    dst_path = os.path.join(ROOT_DIR, label, filename)

    shutil.move(src_path, dst_path)
    print(f"[{label}] {filename} 이동됨")

    # 목록 갱신
    del image_files[current_index]
    if not image_files:  # 다 라벨링했을 때
        messagebox.showinfo("완료", "모든 이미지를 라벨링했습니다.")
        root.quit()
        return

    if current_index >= len(image_files):
        current_index = len(image_files) - 1

    load_image(current_index)


def prev_image(event=None):
    """이전 이미지 보기 (라벨링 없음)"""
    global current_index
    if current_index > 0:
        load_image(current_index - 1)


def next_image(event=None):
    """다음 이미지 보기 (라벨링 없음)"""
    global current_index
    if current_index < len(image_files) - 1:
        load_image(current_index + 1)


# === 버튼 생성 (우측) ===
button_frame = tk.Frame(root)
button_frame.grid(row=0, column=1, sticky="ns", padx=10, pady=10)

cols = 3  # 버튼을 배치할 열 개수

for i, (folder, text) in enumerate(LABELS.items()):
    r = i // cols   # 행
    c = i % cols    # 열
    btn = tk.Button(button_frame, text=text, width=15, height=1,
                    command=lambda f=folder: move_image(f))
    btn.grid(row=r, column=c, padx=5, pady=5, sticky="ew")


# === 방향키 바인딩 (라벨링 없음) ===
root.bind("<Left>", prev_image)
root.bind("<Right>", next_image)

# === 첫 이미지 로드 ===
if image_files:
    load_image(current_index)
else:
    messagebox.showwarning("경고", "라벨링할 이미지가 없습니다.")
    root.quit()

root.mainloop()
