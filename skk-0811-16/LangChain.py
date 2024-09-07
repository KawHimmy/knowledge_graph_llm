from langchain.llms import OpenAI, ChatGLM
import os
from langchain.chains import RetrievalQA
import gradio as gr
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter


# llm = ChatOpenAI(openai_api_key="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")
# model_path = '' GLM的模型路径
# 加载数据，因为加载下来的文件以及是拆分过的，所以不在进行拆分了


embedding_model_dict = {
    'ernie-tiny': "nghuyong/ernie-3.0-nano-zh",
    'text2vec': 'GanymedeNil/text2vec-large-chinese'
}
def load_data(csv_path='./YunnanTravelKG.csv'):
    loader = CSVLoader(file_path=csv_path, encoding='utf-8')
    data = loader.load()
    return data
    pass


# 加载embedding模型
def load_embedding_model(model_name='ernie-tiny'):
    # 如果显存足够可以使用：text2vec3
    # 指定编码和模型参数
    encode_kwargs = {'normalize_embeddings': False}
    model_kwargs = {'device': "cuda:0"}

    # 创建HugggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=embedding_model_dict[model_name],
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    pass


# 创建Chroma实例，将输入的文档数据使用指定的嵌入模型转换为嵌入向量，并将这些向量存储到指定的目录中，以便后续的检索和分析。
def store_chrome(doc, embedings, persist_directory='VectorStore'):
    db = Chroma.from_documents(doc, embedings, persist_directory=persist_directory)
    db.persist()
    return db


def chat(question, history):
    response = qa.run(question)
    return response


if __name__ == "__main__":


    # 断电URL和其他参数初始化
    endpoint_url = "http://0.0.0.0:8000"
    llm = ChatGLM(
        endpoint_url=endpoint_url,
        max_token=80000,
        top_p=0.9
    )

    embeddings = load_embedding_model()
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
        response = qa.run(input_text)
        print(response)

