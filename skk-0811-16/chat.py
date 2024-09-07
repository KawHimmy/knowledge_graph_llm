import os
import requests
from langchain.vectorstores.chroma import Chroma
from LangChain import load_embedding_model
from LangChain import store_chrome, load_data


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
embeddings = load_embedding_model()
if not os.path.exists('VectorStore'):
    data = load_data()
    db = store_chrome(data, embedings=embeddings)
else:
    db = Chroma(persist_directory='VectorStore', embedding_function=embeddings)

while True:
    query = input("your:")
    if query == "quit":
        break
    # include_metadata:给出文件信息, similar_doc:问题答案所在的文档
    similar_docs = db.similarity_search(query, include_metadata=True, k=4)
    prompt = "请回答我的问题，如果你不知道，回答不了，就回复不知道，不要重复我的问题。\n下面是我的相关资料："
    for idx, doc in enumerate(similar_docs):
        prompt += f"{idx + 1}.{doc.page_content}"
    prompt += f"\n下面是问题：{query}"
    response, _ = chat(prompt, [])
    print("Bot:", response)
