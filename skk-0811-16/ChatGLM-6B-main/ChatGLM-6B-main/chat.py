import os
import requests
from langchain.vectorstores.chroma import Chroma
from test import load_embeddings_model
from test import store_chrome, load_data


def chat(prompt, history=None):
    payload = {
        "prompt": prompt, "history": [] if not history else history
    }
    headers = {"Content-Type": "application/json"}

    resp = requests.post(
        url='http://0.0.0.0:8000',
        json=payload,
        headers=headers
    ).json()
    return resp['response'], resp['history']


# 加载embedding模型
embeddings = load_embeddings_model()
if not os.path.exists('VectorStore'):
    data = load_data()
    db = store_chrome(data, embedings=embeddings)
else:
    db = Chroma(persist_directory='VectorStore', embedding_function=embeddings)
wen = []
answer = []
while True:
    inputs = input("your:")
    if inputs == "quit":
        break
    # include_metadata:给出文件信息, similar_doc:问题答案所在的文档
    similar_docs = db.similarity_search(inputs,  k=4)
    if len(wen)==0:
        prompt = "请回答我的问题，如果你不知道，回答不了，就回复不知道，不要重复我的问题。\n下面是我的相关资料："
    else:
        prompt = "上一次我的提问："+wen[0]+"\n上一次你的回答:"+answer[0]+"\n根据以上再回答我的问题，如果你不知道，回答不了，就回复不知道，不要重复我的问题。\n下面是我的相关资料"
        wen=[]
        answer=[]
    for idx, doc in enumerate(similar_docs):
        prompt += f"{idx + 1}.{doc.page_content}"
    prompt += f"\n下面是问题：{inputs}"
    response, _ = chat(prompt, [])
    print("Bot:", response)
    answer.append(response)
    wen.append(inputs)
