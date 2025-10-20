#https://www.youtube.com/watch?v=UtSSMs6ObqY 에서 10:48
#가상환경:moodapi_g(ubuntu)
import ollama
client= ollama.Client()
model = "gemma3:4b"
with open("/mnt/d/non_crop/gemma/test_image.jpg", "rb") as f:#ubuntu라 파일 경로 주의
    image_data = f.read()

response = client.generate(
    model=model,
    prompt="describe this image",
    images=[image_data],  # 핵심: 이미지 데이터를 리스트로 전달
)
print("response from ollama:")
print(response['response'])#response.response는 텍스트만 있을 때
'''
#경로 확인용 디버깅 코드
import os

# 현재 작업 디렉토리 출력
print("현재 작업 디렉토리:", os.getcwd())

# 경로 확인
file_path = "test_image.jpg"
print("파일 경로 확인:", os.path.abspath(file_path))  # 절대 경로 출력

# 경로 존재 여부 확인
if os.path.exists(file_path):
    print(f"파일이 존재합니다: {file_path}")
else:
    print(f"파일을 찾을 수 없습니다: {file_path}")
'''
