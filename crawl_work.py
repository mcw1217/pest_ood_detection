from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
import urllib.request
import os
#식물 일러스트
search='https://www.google.co.kr/search?q=%EC%8B%9D%EB%AC%BC+%EC%9D%BC%EB%9F%AC%EC%8A%A4%ED%8A%B8&sca_esv=04a013b6530774b2&hl=ko&sxsrf=AE3TifOSNJHPnuTUA7ceXI1xBLvQh8pcAA:1760500181997&source=hp&biw=1346&bih=382&ei=1RnvaMH_OpqYwbkPo_rQ4AM&iflsig=AOw8s4IAAAAAaO8n5VJrG-QVrlheM3-slrJopp8XPtdj&ved=0ahUKEwiBvuSfpqWQAxUaTDABHSM9FDwQ4dUDCBc&uact=5&oq=%EC%8B%9D%EB%AC%BC+%EC%9D%BC%EB%9F%AC%EC%8A%A4%ED%8A%B8&gs_lp=EgNpbWciE-yLneusvCDsnbzrn6zsiqTtirgyBxAjGCcYyQIyBxAjGCcYyQIyCBAAGIAEGLEDMgUQABiABDIEEAAYAzIEEAAYAzIFEAAYgAQyBRAAGIAEMgQQABgDMgUQABiABEjRG1AAWLkZcAB4AJABAJgBxwWgAZohqgEDNi02uAEDyAEA-AEBigILZ3dzLXdpei1pbWeYAgKgApALwgILEAAYgAQYsQMYgwGYAwCSBwM1LTKgB5oksgcDNS0yuAeQC8IHBTAuMS4xyAcI&sclient=img&udm=2'

driver = webdriver.Chrome() 
driver.get(search) 
#5. 스크롤 내리기 및 이미지 수집 
SCROLL_PAUSE_TIME = 3 # 스크롤 간 대기 시간
last_height = driver.execute_script('return document.body.scrollHeight') 
# 스크롤을 최대로 내리기 
driver.execute_script('window.scrollTo(0, document.body.scrollHeight);') 
time.sleep(SCROLL_PAUSE_TIME) 
# 새로운 높이를 가져와서 이전 높이와 비교 
new_height = driver.execute_script('return document.body.scrollHeight') 
'''
if new_height == last_height: 
    driver.find_element(By.CSS_SELECTOR, ".mye4qd").click()
''' 
last_height = new_height 
time.sleep(SCROLL_PAUSE_TIME) 
# ✅ 모든 <a href=...> 제거 (탭 열림 방지용) 
driver.execute_script("""
    let anchors = document.querySelectorAll('a[href]');
    anchors.forEach(anchor => anchor.removeAttribute('href'));
""")
# 이미지를 찾고 링크를 links 리스트에 추가 
thumbnails = driver.find_elements(By.CSS_SELECTOR, ".YQ4gaf") 
print(f"썸네일 수: {len(thumbnails)}") 
links=[] 
for thumbnail in thumbnails: 
    try: 
        thumbnail.click()
        # 이미지 로딩 기다리기 
        wait = WebDriverWait(driver, 5)#WebDriverWait으로 최대 5초간 이미지가 로딩될 때까지 기다립 
        img_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".sFlh5c.FyHeAf.iPVvYb"))) 
        # 가능한 모든 img 태그 수집 
        imgs = driver.find_elements(By.CSS_SELECTOR, ".sFlh5c.FyHeAf.iPVvYb") 
        for img in imgs: 
            src = img.get_attribute("src") 
            if src and "encrypted" not in src and not src.startswith("data:"):#and src.startswith("http") 
                if src not in links: 
                    links.append(src) 
                    break 
        time.sleep(1) 
    except Exception as e: 
        print(f"에러: {e}") 
        continue

import os
from urllib.parse import urlparse, unquote
import urllib.request

# 이미지 다운로드
for url in links:
    try:
        print(f"Downloading {url}")
        path = urlparse(url).path
        filename = os.path.basename(path)
        filename = filename.split('?')[0].split('!')[0]
        filename = unquote(filename)

        # 확장자 확인
        name, ext = os.path.splitext(filename)
        if ext.lower() == '.jpg':
            save_filename = filename
        else:
            save_filename = name + '.jpg'

        save_path = f'./그림 크롤링/식물 일러스트04/{save_filename}'

        # 디렉토리 없으면 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/141.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            #"Referer": "https://img.pikbest.com/png-images/20250509/cute-potted-plant-with-smiling-face-transparent-background-for-fun-designs_11706995.png!w700wp",   # 실제 다운로드를 누른 페이지 주소로 바꿔주세요
        }

        # Request 객체 생성
        req = urllib.request.Request(url, headers=headers)

        # 응답 받아서 저장
        with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"이미지가 성공적으로 저장되었습니다: {save_path}")

    except Exception as e:
        print(f'다운로드 실패: {str(e)}')
        continue