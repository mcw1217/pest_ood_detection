#https://www.youtube.com/watch?v=UtSSMs6ObqY 에서 10:48
#가상환경:moodapi_g
import ollama
client= ollama.Client()
model = "gemma3:4b"
prompt = "D:/non_crop/gemma/test_image.jpg describe this image"#질문 넣기
response=client.generate(model=model, prompt=prompt)
print("response from ollama:")
print(response.response)
