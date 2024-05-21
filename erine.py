import time

import requests
import json

from util.db_util import get_it_item_index_id_item_id_mapping, save_llm_answer, save_llm_hint, get_llm_hint, get_llm_answer
from util.file_util import read_file_content_as_string, read_file_content
from util.answer_util import extract_answer_from_str_qianwen

import requests
import json



def get_access_token():
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """

    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=6XuWFXseZVYInmOT2tYShTYn&client_secret=FuKZVXM37bs60fQ8wFK71iOMmMypsCp4"

    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def call_with_prompt(request, access_token):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=" + access_token

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": request
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    return response.text


if __name__ == '__main__':
    access_token = get_access_token()
    it_item_index_id_mapping, id_item_id_map = get_it_item_index_id_item_id_mapping()
    total_score = 0
    for i in range(len(it_item_index_id_mapping)):
        time.sleep(1.1)
        print("index: ", i + 1, " of total: ", len(it_item_index_id_mapping))
        index = it_item_index_id_mapping[i + 1]
        item_id = id_item_id_map[index]
        content = read_file_content_as_string(index, ['content'], True)
        new_hint = read_file_content_as_string(index, ['new_hint'], True)
        answer = read_file_content_as_string(index, ['answer'], True)
        request = '<回答选择题><题干>:' + content + '<回答请用"答案是A"的格式>'
        response = call_with_prompt(request, access_token)

        llm_original_answer = json.dumps(response)
        text = json.loads(response)["result"]

        original_answer = get_llm_answer('ernie-3.5-8k', item_id)
        if len(original_answer) != 0:
            continue

        extracted_answer = extract_answer_from_str_qianwen(text)
        if extracted_answer is None:
            extracted_answer = text
            if len(extracted_answer) > 100:
                extracted_answer = extracted_answer[:100]
        bot_answer = []
        if 'A' in extracted_answer:
            bot_answer.append('A')
        if 'B' in extracted_answer:
            bot_answer.append('B')
        if 'C' in extracted_answer:
            bot_answer.append('C')
        if 'D' in extracted_answer:
            bot_answer.append('D')
        print('bot answer: ', bot_answer)
        llm_answer_json = json.dumps(bot_answer)
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
        true_answer_json = json.dumps(true_answer)

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

        total_score += score
        print("current average score: ", total_score / (i + 1))

        save_llm_answer('ernie-3.5-8k',
                        llm_original_answer,
                        answer,
                        llm_answer_json,
                        true_answer_json,
                        score,
                        item_id)
