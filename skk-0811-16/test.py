from pydantic import BaseModel
from typing import List
import requests
import os
import json
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    dialog: List[Message]
    do_sample: bool = True
    # max_new_tokens: int = 50000
    max_length: int = 15000
    top_k: int = 3
    top_p: float = 0.2
    temperature: float = 0.1
    repetition_penalty: float = 1.03

API_URL = "http://localhost:2024/query"

def chat(dialog, do_sample=True, max_length=8000, top_k=3, top_p=0.2, temperature=0.1, repetition_penalty=1.03):
    payload = QueryRequest(
        dialog=dialog,
        do_sample=do_sample,
        max_length=max_length,  
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty
    ).dict()
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(
            url=API_URL,
            json=payload,
            headers=headers
        )
        resp.raise_for_status()  # 检查请求是否成功
        data = resp.json()

        # # 打印响应内容以便调试
        # print("Response data:", data)

        # 确保返回的响应中有 'answer' 和 'history' 键
        response = data.get('answer')
        history = data.get('history')

#         if response is None or history is None:
#             raise KeyError("Missing 'answer' or 'history' in response data.")

        return response, history
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        print(f"Response content: {e.response.text}")
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None, None
    except KeyError as e:
        print(f"KeyError: {e}")
        return None, None

# # 测试代码
# dialog = [
#     {"role": "user", "content": "Hello, how are you?"}
# ]
# response, history = chat(dialog)
# print("Response:", response)
# print("History:", history)