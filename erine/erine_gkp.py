import random
import time

import numpy as np
import requests
import json

from bs4 import BeautifulSoup
from tqdm import tqdm

from util.db_util import get_it_item_index_id_item_id_mapping, save_llm_answer, save_llm_hint, get_llm_hint, get_llm_answer
from util.file_util import read_file_content_as_string, read_file_content
from util.answer_util import extract_answer_from_str_qianwen

import requests
import json
import mysql.connector

def open_database():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="Lentopaz35628",
        database="ayesha",
        auth_plugin='mysql_native_password'
    )


def select_dataset(sql):
    db = open_database()
    dataset = []
    cursor = db.cursor()
    cursor.execute(sql)
    results = cursor.fetchall()
    for i in range(len(results)):
        dataset.append(results[i])
    cursor.close()
    db.close()
    return dataset


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

    while True:
        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.text)
        if "result" in json.loads(response.text):
            break
        time.sleep(1)
    return response.text

def get_item_content(item_id: int):
    dataset = select_dataset("select * from item_index_16 where item_id = "+str(item_id))
    if len(dataset)==0:
        return None
    item = json.loads(dataset[0][10])
    # 提取信息
    content_html = item['bundler']['content']
    # 去除html
    return BeautifulSoup(content_html, "html.parser").text


# 随机选取5个，由llm生成提示信息append到问题上
if __name__ == '__main__':
    random.seed(20240101)
    knowledge_dataset = select_dataset(
        "select * from knowledge_database where length(hint) > 0;")
    random_knowledge_dataset = random.sample(knowledge_dataset, 5)
    prompt = ""
    for random_knowledge in random_knowledge_dataset:
        item_id = random_knowledge[1]
        hint = random_knowledge[3]
        content = get_item_content(item_id)
        prompt += "题干：" + content + "\n知识：" + hint + "\n"

    access_token = get_access_token()
    dataset = select_dataset(
        "select * from item_index_16 where knowledge = 1 and `ignore` = 0 and is_chinese = 1 and is_empty = 0 and item_type = 1")

    total_score = 0
    for i in range(len(dataset)):
        item = json.loads(dataset[i][10])
        item_id = int(dataset[i][1])
        # 提取信息
        content_html = item['bundler']['content']
        # 去除html
        content_clean = BeautifulSoup(content_html, "html.parser").text

        # 提取信息
        answer_html = item['bundler']['answers']
        # 去除html
        answer_clean = BeautifulSoup(answer_html, "html.parser").text

        knowledge_prompt = prompt + "题干：" + content_clean + "\n知识："
        hint = call_with_prompt(knowledge_prompt, access_token)

        original_answer = get_llm_answer('ernie_gkp', item_id)
        if len(original_answer) != 0:
            continue

        request = hint + "\n" + content_clean + "\n<回答请用'答案是A'的格式>"
        response = call_with_prompt(request, access_token)

        llm_original_answer = json.dumps(response)
        text = json.loads(response)["result"]

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
        if 'A' in answer_clean:
            true_answer.append('A')
        if 'B' in answer_clean:
            true_answer.append('B')
        if 'C' in answer_clean:
            true_answer.append('C')
        if 'D' in answer_clean:
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

        save_llm_answer('ernie_gkp',
                        llm_original_answer,
                        answer_clean,
                        llm_answer_json,
                        true_answer_json,
                        score,
                        item_id)

