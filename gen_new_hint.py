import json
import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import readline
from util.db_util import get_it_item_index_id_mapping
from util.file_util import read_file_content_as_string, index_to_path
from util.config_util import DATA_FOLDER

tokenizer = AutoTokenizer.from_pretrained("/home/len/Desktop/git/ChatGLM-6B/THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/len/Desktop/git/ChatGLM-6B/THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def list_to_file(file_name: str, content: list):
    with open(file_name, 'w', encoding="utf-8") as f:
        for s in content:
            f.write(s + '\n')

if __name__ == "__main__":
    it_item_index_id_mapping = get_it_item_index_id_mapping()

    for i in range(len(it_item_index_id_mapping)):
        index = it_item_index_id_mapping[i + 1]
        content = read_file_content_as_string(index, ['content'], True)
        knowledge = read_file_content_as_string(index, ['top_knowledge'], True)

        request = '根据<题干>:' + content + '<知识点>:' + knowledge + '<生成提示信息>'
        response, history = model.chat(tokenizer, request, history=[])
        response = response.replace(request, '')
        response = response.replace('<题干>', '')
        response = response.replace(content, '')
        response = response.replace(knowledge, '')
        response = response.replace('<知识点>', '')
        response = response.replace('<生成提示信息>', '')
        response = response.replace('\n', '')
        response = response.replace('\n\r', '')
        print('提示信息: ', response)