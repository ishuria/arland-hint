import json
import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import readline
from util.db_util import get_it_item_index_id_mapping
from util.file_util import read_file_content_as_string, index_to_path
from util.config_util import DATA_FOLDER, CHATGLM_6B_FOLDER

tokenizer = AutoTokenizer.from_pretrained(CHATGLM_6B_FOLDER, trust_remote_code=True)
model = AutoModel.from_pretrained(CHATGLM_6B_FOLDER, trust_remote_code=True).half().cuda()
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


train_datas = []
with open('/home/len/information-hint-data/kancd-data.txt', 'r') as f:
    train_datas = [line.rstrip('\n') for line in f]

knowledge_index_map = {}
with open('/home/len/information-hint-data/class.txt', 'r') as f:
    knowledge_list = [line.rstrip('\n') for line in f]
    for i in range(len(knowledge_list)):
        knowledge_index_map[knowledge_list[i]] = i + 1


def main():
    history = []
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    #            print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)


# 每10万个记录形成一个文件夹
# levelOne/levelTwo/levelThree/dividend
def index_to_path(item_index: int):
    levelOne = 0
    levelTwo = 0
    levelThree = 0
    dividend = item_index % 100;
    item_index //= 100
    levelThree = item_index % 100;
    item_index //= 100
    levelTwo = item_index % 100;
    item_index //= 100
    levelOne = item_index % 100;
    item_index //= 100
    return str(levelOne) + os.sep + str(levelTwo) + os.sep + str(levelThree) + os.sep + str(dividend)


def read_file_content(item_index: int, categories: list, no_sep: bool):
    list = []
    for category in categories:
        _list = []
        if not os.path.isfile(DATA_FOLDER + index_to_path(item_index) + os.sep + category + '.txt'):
            continue
        with open(DATA_FOLDER + index_to_path(item_index) + os.sep + category + '.txt', 'r', encoding='utf-8') as f:
            _list = [line.rstrip('\n') for line in f]
        list.extend(_list)
        if not no_sep:
            list.append('<sep>')
    return list

new_hint_map = {}

new_hint_list = []
with open('/home/len/kancd-data/new_hint.txt', 'r', encoding='utf-8') as f:
    new_hint_list = [line.rstrip('\n') for line in f]

for new_hint in new_hint_list:
    id, h = new_hint.split('\t')
    new_hint_map[id]=h

if __name__ == "__main__":
    it_item_index_id_mapping = get_it_item_index_id_mapping()
    score_file = []
    total_score = 0
    for i in range(len(it_item_index_id_mapping)):
        index = it_item_index_id_mapping[i + 1]
        content = read_file_content_as_string(index, ['content'], True)
        new_hint = read_file_content_as_string(index, ['new_hint'], True)
        answer = read_file_content_as_string(index, ['new_hint'], True)
        request = '<回答选择题><题干>:' + content + '<提示>:' + new_hint
        response, history = model.chat(tokenizer, request, history=[])
        # print('content: ', content)
        # print('hint: ', hint)
        # print('knowledge: ', knowledge)
        # response = response.replace(request, '')
        # response = response.replace('<题目>', '')
        # response = response.replace(content, '')
        # response = response.replace(hint, '')
        # response = response.replace(knowledge, '')
        # response = response.replace('<旧提示>', '')
        # response = response.replace('<知识点>', '')
        # response = response.replace('<生成新提示信息>', '')
        # print('new hint: ', response)
        # request = '<回答选择题><题目>:' + content + '<提示>:' + response
        # response, history = model.chat(tokenizer, request, history=[])
        # new_hint_file.append(str(id) + ',' + response)
        # if train_data_index > 500:
        #    break
        if len(response) > 100:
            response = response[-100:]
        bot_answer = []
        if 'A' in response:
            bot_answer.append('A')
        if 'B' in response:
            bot_answer.append('B')
        if 'C' in response:
            bot_answer.append('C')
        if 'D' in response:
            bot_answer.append('D')
        print('bot answer: ', bot_answer)
        true_answer = []
        if 'A' in answer:
            true_answer.append('A')
        if 'B' in answer:
            true_answer.append('B')
        if 'C' in answer:
            true_answer.append('C')
        if 'D' in answer:
            true_answer.append('D')
        print('true answer: ', true_answer)

        max_length = len(bot_answer)
        if len(true_answer) > len(bot_answer):
            max_length = len(true_answer)

        match_count = 0
        for ta in true_answer:
            for ba in bot_answer:
                if ba == ta:
                    match_count += 1
                    break
        if len(true_answer) == 0:
            continue

        score = round(float(match_count) / float(len(true_answer)), 2)
        if len(bot_answer) > len(true_answer):
            score = 0.0

        print('score: ', score)

        knowledge_str_list = read_file_content(int(id), ['knowledge', 'auto_knowledge'], True)
        knowledge_id_list = []
        for knowledge_str in knowledge_str_list:
            knowledge_id_list.append(knowledge_index_map[knowledge_str])
        # 写item
        # item_file.append(str(id) + ',' + json.dumps(knowledge_id_list))
        score_file.append('1,' + str(id) + ',' + str(score))

        total_score += score
        print("current average score: " ,total_score / (i+1))



    # list_to_file('/home/len/kancd-data/with_new_hint/item.csv', item_file)
    # list_to_file('/home/len/kancd-data/with_new_hint/train.csv', score_file)