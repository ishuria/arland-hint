import random
import re
import time
from bs4 import BeautifulSoup
import requests
import json
import mysql.connector
from scipy.special import softmax
import numpy as np


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
        """, {
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
    cursor.execute(
        "select * from llm_answer where item_id = " + str(item_id) + " and llm_name = '" + str(llm_name) + "'")
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
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            print(response.text)
            if "result" in json.loads(response.text):
                break
            time.sleep(1)
        except Exception as e:
            print(e)
    return response.text


def get_item_content(item_id: int):
    dataset = select_dataset("select * from item_index_16 where item_id = " + str(item_id))
    if len(dataset) == 0:
        return None
    item = json.loads(dataset[0][10])
    # 提取信息
    content_html = item['bundler']['content']
    # 去除html
    return BeautifulSoup(content_html, "html.parser").text


def get_knowledge_hint(knowledge: str):
    return select_dataset("select * from knowledge_database where knowledge = '" + knowledge + "'")


# 随机选取5个，由llm生成提示信息append到问题上
if __name__ == '__main__':
    random.seed(20240101)
    knowledge_dataset = select_dataset(
        "select * from knowledge_database where length(hint) > 0;")

    access_token = get_access_token()
    dataset = select_dataset(
        "select * from item_index_16 where knowledge = 1 and `ignore` = 0 and is_chinese = 1 and is_empty = 0 and item_type = 1")

    knowledge_cd_str = ''
    with open('../answer/erine.cd.txt', 'r', encoding="utf-8") as f:
        for line in f:
            knowledge_cd_str += line.rstrip('\n')
    knowledge_cd = json.loads(knowledge_cd_str)

    knowledge_id_name_dict = {}
    with open('../answer/item.knowledge.name', 'r', encoding="utf-8") as f:
        for line in f:
            elements = line.rstrip('\n').split('\t')
            knowledge_id_name_dict[int(elements[1])] = elements[0]

    item_features = {}
    with open('../answer/item.content', 'r', encoding="utf-8") as f:
        for line in f:
            feature = []
            elements = line.rstrip('\n').split('\t')
            for i in range(len(elements)):
                if i == 0:
                    continue
                if i == len(elements) - 1:
                    continue
                feature.append(int(elements[i]))
            item_features[elements[0]] = feature

    knowledge_cd = softmax(knowledge_cd)

    total_score = 0
    for i in range(len(dataset)):
        print("index: ", i + 1, " of total: ", len(dataset))
        item = json.loads(dataset[i][10])
        item_id = int(dataset[i][1])
        original_answer = get_llm_answer('erine_mlgakt', item_id)
        if len(original_answer) != 0:
            continue
        std_features = softmax(item_features[str(item_id)])
        min_knowledge_cd = [x - float(y) for (x, y) in zip(knowledge_cd, std_features)]
        min_idxes = np.argsort(min_knowledge_cd)[:5]

        prompt = ""
        for min_idx in min_idxes:
            knowledge = knowledge_id_name_dict[min_idx + 1]
            knowledge_hints = get_knowledge_hint(knowledge)
            if len(knowledge_hints) == 0:
                # 随机选取1个
                random_knowledge_dataset = random.sample(knowledge_dataset, 1)
                for random_knowledge in random_knowledge_dataset:
                    item_id = random_knowledge[1]
                    hint = random_knowledge[3]
                    content = get_item_content(item_id)
                    prompt += "题干：" + content + "\n知识：" + hint + "\n"
            else:
                random_knowledge_dataset = random.sample(knowledge_hints, 1)
                for random_knowledge in random_knowledge_dataset:
                    item_id = random_knowledge[1]
                    hint = random_knowledge[3]
                    content = get_item_content(item_id)
                    prompt += "题干：" + content + "\n知识：" + hint + "\n"

        # 提取信息
        content_html = item['bundler']['content']
        # 去除html
        content_clean = BeautifulSoup(content_html, "html.parser").text

        # 提取信息
        answer_html = item['bundler']['answers']
        # 去除html
        answer_clean = BeautifulSoup(answer_html, "html.parser").text

        knowledge_prompt = prompt + "题干：" + content_clean + "\n知识："
        hint = json.loads(call_with_prompt(knowledge_prompt, access_token))["result"]

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

        save_llm_answer('erine_mlgakt',
                        llm_original_answer,
                        answer_clean,
                        llm_answer_json,
                        true_answer_json,
                        score,
                        item_id)
