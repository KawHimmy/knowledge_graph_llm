from googletrans import Translator  
  
def translate_text(text, src='en', dest='zh-cn'):  
    translator = Translator()  
    result = translator.translate(text, src=src, dest=dest)  
    return result.text  
  
text = "Hello, how are you?"  
translated_text = translate_text(text)  
print(translated_text)