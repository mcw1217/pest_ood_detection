# main.py
from load_dataset import load_dataset

# 사용자 입력 받기
name = input("사용할 데이터셋 이름을 입력하세요 (ninco / openimageo / texture): ")

# 데이터셋 로드
dataset = load_dataset(name)

# 결과 출력
print(f"{name} 데이터셋 로드 완료! 이미지 개수: {len(dataset)}")
