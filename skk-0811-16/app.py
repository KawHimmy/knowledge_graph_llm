from flask import Flask, render_template, request, redirect, url_for,jsonify
import csv
import random
import torch
import torch.nn.functional as F
import numpy as np
import dataloader4kg
from kgcn import KGCN
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests
import random
import os
import requests
from langchain.vectorstores.chroma import Chroma
from test import load_embeddings_model
from test import store_chrome, load_data
import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True




# 加载embedding模型
embeddings = load_embeddings_model()
if not os.path.exists('VectorStore'):
    data = load_data()
    db = store_chrome(data, embedings=embeddings)
else:
    db = Chroma(persist_directory='VectorStore', embedding_function=embeddings)

app = Flask(__name__, static_folder='templates/static')




def my_chat(prompt, history=None):
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
class TextRecommendation:
    def __init__(self, data_file_path='skk-0811-16/文本主题关注度.csv'):
        self.df = pd.read_csv(data_file_path)

    def recommend_texts(self, user_interests):
        # 提取文本的主题相关度作为文本向量
        text_vectors = self.df[[f'topic_{i}' for i in range(len(user_interests))]].values

        # 构建用户关注度向量
        user_vector = [user_interests[f'topic_{i}'] for i in range(len(user_interests))]

        # 计算余弦相似度
        self.df['cosine_similarity'] = cosine_similarity(text_vectors, [user_vector]).flatten()

        # 根据余弦相似度排序，取得分最高的两个文本
        top_texts = self.df.sort_values(by='cosine_similarity', ascending=False).head(1)

        return top_texts['全文']

users = {'kawhi': '123456', 'mama': '666666'}
path = "skk-0811-16/data/encoded_entities.tsv"

class KG:
    def __init__(self, model_path, entity_mapping_path, device="cuda:0" if torch.cuda.is_available() else'cpu'):
        self.device = torch.device(device)
        self.model_path = model_path
        self.entity_mapping_path = entity_mapping_path
        self.model = None
        self.entity_id_to_name = {}
        self.new_set = None
        self.users = None

    def load_model(self):
        with open(self.entity_mapping_path, 'r', encoding='utf-8') as file:
            for line in file:
                name, id_str = line.strip().split('\t')
                entity_id = int(id_str)
                self.entity_id_to_name[entity_id] = name
                # 创建并加载模型
        self.users, items, train_set, test_set, self.new_set = dataloader4kg.read_ClickData(dataloader4kg.Travel_yun.RATING, dataloader4kg.Travel_yun.rating1)

        entitys, relations, kgTriples = dataloader4kg.read_KG(dataloader4kg.Travel_yun.KG)
        adj_kg = dataloader4kg.construct_kg(kgTriples)
        adj_entity, adj_relation = dataloader4kg.construct_adj(10, adj_kg, len(entitys))
        # 将邻接矩阵转移到GPU上
        adj_entity = torch.LongTensor(adj_entity).to(self.device)
        adj_relation = torch.LongTensor(adj_relation).to(self.device)
        self.model = KGCN(max(self.users) + 1, n_entitys=max(entitys) + 1, n_relations=max(relations) + 1,
                     e_dim=10, adj_entity=adj_entity, adj_relation=adj_relation,
                     n_neighbors=10,
                     aggregator_method='sum',
                     act_method=F.relu,
                     drop_rate=0.5).to(self.device)
        return self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
    
    def recommend_for_user(self, top_k=10):
        new_set = torch.LongTensor(self.new_set)
        self.model.eval()
        with torch.no_grad():
            user_ids = new_set[:, 0].to(self.device)
            item_ids = new_set[:, 1].to(self.device)
            logits = self.model(user_ids, item_ids, True)
            predictions = [1 if i >= 0.5 else 0 for i in logits]

            top_k_items = np.argsort(predictions)[::-1][:top_k]

            recommended_entity_names = [self.entity_id_to_name[entity_id] for entity_id in top_k_items]

            return  recommended_entity_names

@app.route('/', methods=['GET', 'POST'])
def login():
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username not in users:
            users[username] = password
            return render_template('login.html')
        else:
            return 'Username already exists. Please choose a different username.'

    return render_template('register.html')

@app.route('/chat.html', methods=['GET', 'POST'])
def chat():
    print("555")
    return render_template('chat.html')

@app.route('/penson.html', methods=['GET', 'POST'])
def penson():
    return render_template('penson.html')

@app.route('/map.html', methods=['GET', 'POST'])
def map():
    return render_template('map.html')

@app.route('/index.html', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/contact.html', methods=['GET', 'POST'],endpoint = 'contact_page')
def index():
    return render_template('contact.html')

@app.route('/tag.html', methods=['GET', 'POST'])
def tag():
    with open(path, 'r', newline='', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        # 读取文件内容
        all_rows = list(tsv_reader)
        all_list=[]
        # 随机抽取三条数据
        random_rows = random.sample(all_rows, 3)
        for row in random_rows:
            all_list.append(row[0])
    button_names = all_list
    # 将按钮名字传递到模板
    return render_template('tag.html', button_names=button_names)
temp = 0
@app.route('/get_random_words', methods=['GET','POST'])
def get_random_words():
    with open(path, 'r', newline='', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        # 读取文件内容
        all_rows = list(tsv_reader)
        all_list=[]
        nums_list=[]
        # 随机抽取三条数据
        random_rows = random.sample(all_rows, 3)
        for row in random_rows:
            all_list.append(row[0])
            nums_list.append(row[1])
    new_button_names = all_list
    button_index = (request.form.get('buttonIndex'))
    global temp
    temp = button_index
    return jsonify(new_button_names)
@app.route('/get_button_images')
def get_button_images():
    # 假设你有一个包含图片 URL 的列表
    button_images = ["static/image/1.jpg", "static/image/2.jpg", "static/image/3.jpg", "static/image/4.jpg"]
    for i in range(5,102):
        button_images.append("static/image/" + str(i) + ".jpg")
    return jsonify(button_images)
@app.route('/commadation',methods=['GET', 'POST'])
def commadation():
    with open(path, 'r', newline='', encoding='utf-8') as file:
        tsv_reader = csv.reader(file, delimiter='\t')
        # 读取文件内容
        all_rows = list(tsv_reader)
        all_list=[]
        nums_list=[]
        # 随机抽取三条数据
        random_rows = random.sample(all_rows, 3)
        for row in random_rows:
            all_list.append(row[0])
            nums_list.append(row[1])
    new_button_names = all_list
    aaa = temp
    print(aaa)
    if aaa is not None:
            print(22222222222)
            ans=[]
            # print(nums_list)
            for jj in range(3):
                if jj == int(aaa):
                    ans.append(str(1428))
                    ans.append(nums_list[jj])
                    ans.append(str(1))
                    with open("skk-0811-16/data/new_click.tsv", 'a', newline='', encoding='utf-8') as file:
                        tsv_writter = csv.writer(file,delimiter='\t')
                        tsv_writter.writerow(ans)
                        ans = []
                else:
                    ans.append(str(1428))
                    ans.append(nums_list[jj])
                    ans.append(str(0))
                    with open("skk-0811-16/data/new_click.tsv", 'a', newline='', encoding='utf-8') as file:
                        tsv_writter = csv.writer(file,delimiter='\t')
                        tsv_writter.writerow(ans)
                        ans = []
            n_neighbors = 10
            model_path = 'skk-0811-16/model/model.pth'
            entity_mapping_path = 'skk-0811-16/data/encoded_entities.tsv'

            recommendation_model = KG(model_path, entity_mapping_path)

            # 加载模型
            recommendation_model.load_model()

            top_k_recommendations = recommendation_model.recommend_for_user(top_k=n_neighbors)
            print("成功了")
            return jsonify(top_k_recommendations=top_k_recommendations)
    print("Returning default response")
    return jsonify({})




@app.route('/tuijian_gonglue',methods = ['POST'])
def tuijian_gonglue():
    text_recommendation = TextRecommendation()
    # 假设用户对每个主题的关注度
    user_interests = {'topic_0': random.random(), 'topic_1': random.random(), 'topic_2': random.random(),
                   'topic_3': random.random(), 'topic_4': random.random()}

# 计算总和
    total_weight = sum(user_interests.values())

# 归一化处理，使总和为1.0
    user_interests = {key: value / total_weight for key, value in user_interests.items()}

    print(user_interests)
    recommended_texts = text_recommendation.recommend_texts(user_interests).to_json()

    import codecs
    decoded_str = codecs.decode(recommended_texts, 'unicode_escape')
    decoded_dict = json.loads(decoded_str)
    values = list(decoded_dict.values())
    temp  = values[0]
    count = 0
    result = '     '
    for char in temp:
        if char == ',':
            count += 1
            if count % 30 ==0:
                char = "。\n     "
                count = 0
        result += char
    values = [result]
    print(values)
    return jsonify(recommended_texts = values)
wen = []
answer = []
@app.route('/process_input', methods=['POST'])
def process_input():
    global wen
    global answer
    global stop_stream
    history = []
    data = request.get_json()
    inputs = data['input']
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
    count = 0
    for response, history in model.stream_chat(tokenizer, prompt, history=history):
        if stop_stream:
            stop_stream = False
            break
        else:
            count += 1
            if count % 8 == 0:
                os.system(clear_command)
                print(build_prompt(history), flush=True)
    os.system(clear_command)
    query = build_prompt(history)
    substring = "ChatGLM-6B："
    index = query.find(substring)
    if index != -1:
        query = query[index + len(substring):]
    answer.append(response)
    wen.append(inputs)
    return jsonify({'output': query})

if __name__ == '__main__':
    app.run(debug=True)

