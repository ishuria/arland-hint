import json
from random import random
import re
import jieba
import mysql.connector
from tqdm import tqdm
from stopwds import stopwords
from bs4 import BeautifulSoup


def open_database():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="123456",
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


def save_to_knowledge_base(knowledge: str,
                           hint: str,
                           item_id: int):
    db = open_database()
    db.set_charset_collation('utf8mb4')
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO `ayesha`.`knowledge_database`
        (`knowledge`,
        `hint`,
        `item_id`)
        VALUES
        (%(knowledge)s,
        %(hint)s,
        %(item_id)s);
        """, {
        "knowledge": knowledge,
        "hint": hint,
        "item_id": item_id
    })
    cursor.close()
    db.commit()
    db.close()


def update_filtered_hint(filtered_hint: str,
                         id: int):
    db = open_database()
    db.set_charset_collation('utf8mb4')
    cursor = db.cursor()
    cursor.execute("""
        UPDATE `ayesha`.`knowledge_database` set filtered_hint=%(filtered_hint)s where id=%(id)s;
        """, {
        "filtered_hint": filtered_hint,
        "id": id
    })
    cursor.close()
    db.commit()
    db.close()


def print_hint():
    dataset = select_dataset(
        "select * from item_index_16 where knowledge = 1 and `ignore` = 0 and is_chinese = 1 and is_empty = 0 and item_type = 1")
    item_knowledge_dict = {}
    for data in tqdm(dataset, "build_knowledge_id_file"):
        item = json.loads(data[10])
        knowledge_name_set = set()
        for point in item["points"]:
            knowledge_name_set.add(point["name"])
        item_knowledge_dict[item["item_id"]] = knowledge_name_set

        # 提取信息
        content_html = item['bundler']['content']
        # 去除html
        content_clean = BeautifulSoup(content_html, "html.parser").text
        # # 分词
        # content_split = list(jieba.cut(content_clean))
        # content_split = [x for x in content_split if len(x) > 1 and x not in stopwords()]

        # 提取信息
        hint_html = item['bundler']['hint']
        # 去除html
        hint_clean = BeautifulSoup(hint_html, "html.parser").text

        if len(hint_clean.strip()) > 0 and hint_clean.strip().find("略") < 0:
            print("content_clean: ", content_clean)
            print("hint_clean: ", hint_clean)
            print("item_knowledge_dict: ", item_knowledge_dict[item["item_id"]])
            print("\n")

            for knowledge in item_knowledge_dict[item["item_id"]]:
                save_to_knowledge_base(knowledge, hint_clean, item["item_id"])


def filter_hint():
    dataset = select_dataset("select id, hint from knowledge_database;")
    regex = r"本题.*?。(.*?)故.*?。"
    for data in tqdm(dataset, "filter_hint"):
        id = data[0]
        hint = data[1]
        filtered_hint = ""
        matches = re.finditer(regex, hint, re.MULTILINE)
        for matchNum, match in enumerate(matches, start=1):
            for groupNum in range(0, len(match.groups())):
                filtered_hint = match.group(groupNum + 1)
                break
            break
        print(filtered_hint)
        update_filtered_hint(filtered_hint, id)


if __name__ == "__main__":
    print_hint()
    # filter_hint()
