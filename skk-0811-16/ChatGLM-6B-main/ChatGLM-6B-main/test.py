from langchain.llms import OpenAI, ChatGLM
import os
from langchain.chains import RetrievalQA
import gradio as gr
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from googletrans import Translator 
embedding_model_dict = {
"ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
"ernie-base": "nghuyong/ernie-3.0-base-zh",
"text2vec": "GanymedeNil/text2vec-large-chinese",
"text2vec2":"uer/sbert-base -chinese-nli",
"text2vec3": "shibing624/text2vec-base-chinese",
}
def load_data(csv_path='KG_.csv'):
    loader = CSVLoader(file_path=csv_path, encoding='utf-8')
    data = loader.load()
    return data
    pass
def load_embeddings_model(model_name = "ernie-tiny"):
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}
    return HuggingFaceEmbeddings(
        model_name = embedding_model_dict[model_name],
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )
def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db
embeddings = load_embeddings_model("text2vec3")
# 创建Chroma实例，将输入的文档数据使用指定的嵌入模型转换为嵌入向量，并将这些向量存储到指定的目录中，以便后续的检索和分析。
def store_chrome(doc, embedings, persist_directory='VectorStore'):
    db = Chroma.from_documents(doc, embedings, persist_directory=persist_directory)
    db.persist()
    return db
def translate_text(text, src='en', dest='zh-cn'):  
    translator = Translator()  
    result = translator.translate(text, src=src, dest=dest)  
    return result.text  

def chat(question, history):
    response = qa.run(question)
    return response


if __name__ == "__main__":


    # 断电URL和其他参数初始化
    endpoint_url = "http://0.0.0.0:8070"
    llm = ChatGLM(
        endpoint_url=endpoint_url,
        max_token=80000,
        top_p=0.9
    )

    embeddings = load_embeddings_model()
    # 检查是否有己经处理的数据库
    if not os.path.exists('VectorStore'):
        docu = load_data()
        db = store_chrome(docu, embeddings)
    else:
        db = Chroma(persist_directory='VectorStore', embedding_function=embeddings)

    # 提取文档做QA操作
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever
    )
    while True:
        input_text = input("您的问题 ")
        if input_text.lower() == 'quit':
            break
        input_text = input_text
        response = qa.run(input_text)
        print(response)
