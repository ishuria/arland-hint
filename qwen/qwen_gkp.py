import random
import re
import time


from bs4 import BeautifulSoup

import dashscope
from http import HTTPStatus
import requests
import json
import mysql.connector

def open_database():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="123456",
        database="ayesha",
        auth_plugin='mysql_native_password'
    )

def extract_answer_from_str_qianwen(answer_str: str):
    regex = r"答案是.*?([A-Z]*)|答案是：.*?([A-Z]+)"
    matches = re.finditer(regex, answer_str, re.MULTILINE)
    for matchNum, match in enumerate(matches, start=1):
        # print ("Match {matchNum} was found at {start}-{end}: {match}".format(matchNum = matchNum, start = match.start(), end = match.end(), match = match.group()))
        for groupNum in range(0, len(match.groups())):
            groupNum = groupNum + 1
            if match.group(groupNum) is None:
                continue
            return match.group(groupNum)
            # print ("Group {groupNum} found at {start}-{end}: {group}".format(groupNum = groupNum, start = match.start(groupNum), end = match.end(groupNum), group = match.group(groupNum)))
    return None


def save_llm_answer(llm_name: str,
                    llm_original_answer: str,
                    true_original_answer: str,
                    llm_answer: str,
                    true_answer: str,
                    score: float,
                    item_id: int):
    db = open_database()
    db.set_charset_collation('utf8mb4')
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO `ayesha`.`llm_answer`
        (`llm_name`,
        `llm_original_answer`,
        `true_original_answer`,
        `llm_answer`,
        `true_answer`,
        `score`,
        `item_id`)
        VALUES
        (%(llm_name)s,
        %(llm_original_answer)s,
        %(true_original_answer)s,
        %(llm_answer)s,
        %(true_answer)s,
        %(score)s, 
        %(item_id)s);
        """ ,{
            "llm_name": llm_name,
            "llm_original_answer": llm_original_answer,
            "true_original_answer": true_original_answer,
            "llm_answer": llm_answer,
            "true_answer": true_answer,
            "score": score,
            "item_id": item_id
            })
    cursor.close()
    db.commit()
    db.close()


def get_llm_answer(llm_name: str, item_id: int):
    db = open_database()
    cursor = db.cursor()
    cursor.execute("select * from llm_answer where item_id = " + str(item_id) + " and llm_name = '" + str(llm_name) + "'")
    results = cursor.fetchall()
    original_answer = ""
    for i in range(len(results)):
        original_answer = results[i][2]
    cursor.close()
    db.close()
    return original_answer

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

def call_with_prompt(prompt):
    while True:
        try:
            response = dashscope.Generation.call(
                model=dashscope.Generation.Models.qwen_turbo,
                prompt=prompt
            )
            # 如果调用成功，则打印模型的输出
            if response.status_code == HTTPStatus.OK:
                print(response)
                return response
            # 如果调用失败，则打印出错误码与失败信息
            else:
                print(response.code)
                print(response.message)
        except Exception as e:
            print(e)

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

    dataset = select_dataset(
        "select * from item_index_16 where knowledge = 1 and `ignore` = 0 and is_chinese = 1 and is_empty = 0 and item_type = 1")

    total_score = 0
    for i in range(len(dataset)):
        print("index: ", i + 1, " of total: ", len(dataset))
        item = json.loads(dataset[i][10])
        item_id = int(dataset[i][1])
        original_answer = get_llm_answer('qwen_gkp', item_id)
        if len(original_answer) != 0:
            continue

        # 提取信息
        content_html = item['bundler']['content']
        # 去除html
        content_clean = BeautifulSoup(content_html, "html.parser").text

        # 提取信息
        answer_html = item['bundler']['answers']
        # 去除html
        answer_clean = BeautifulSoup(answer_html, "html.parser").text

        knowledge_prompt = prompt + "题干：" + content_clean + "\n知识："
        hint = call_with_prompt(knowledge_prompt)["output"]["text"]

        request = hint + "\n" + content_clean + "\n<回答请用'答案是A'的格式>"
        response = call_with_prompt(request)

        llm_original_answer = json.dumps(response)
        text = response["output"]["text"]

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

        save_llm_answer('qwen_gkp',
                        llm_original_answer,
                        answer_clean,
                        llm_answer_json,
                        true_answer_json,
                        score,
                        item_id)

