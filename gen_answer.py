import json
import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import readline
from util.db_util import get_it_item_index_id_mapping
from util.file_util import read_file_content_as_string, read_file_content
from util.config_util import DATA_FOLDER, CHATGLM_6B_FOLDER

tokenizer = AutoTokenizer.from_pretrained(CHATGLM_6B_FOLDER, trust_remote_code=True)
model = AutoModel.from_pretrained(CHATGLM_6B_FOLDER, trust_remote_code=True).half().cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

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

        score_file.append('1,' + str(id) + ',' + str(score))

        total_score += score
        print("current average score: " ,total_score / (i+1))



    # list_to_file('/home/len/kancd-data/with_new_hint/item.csv', item_file)
    # list_to_file('/home/len/kancd-data/with_new_hint/train.csv', score_file)