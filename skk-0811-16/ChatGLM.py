
from transformers import AutoTokenizer, AutoModelForCausalLM
#
# # 指定模型和分词器的本地路径
# model_path = "./model-ChatGLM-6B"
# # model_path = 'C:\\Users\\15645\\.cache\\huggingface\\modules\\transformers_modules\\:'
# # 从本地路径加载分词器和模型
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().cuda()
#
# # 确保您的模型有 chat 方法，如果没有，则您需要按照模型的使用说明进行对话生成
# # 下面是一个示例，具体取决于模型的具体实现
# response, history = model.chat(tokenizer, "你好", history=[])
# print(response)
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)


from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4-qe", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4-qe", trust_remote_code=True).half().cuda()
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4-qe", trust_remote_code=True).half().cuda().eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)
while True:
    input_text = input()
    if input_text.lower() == 'quit':
        break
    response, history = model.chat(tokenizer, input_text, history=history)
    print(response)
