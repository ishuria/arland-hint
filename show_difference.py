import mysql.connector
import json
from util.config_util import DB_HOST

if __name__ == "__main__":
    db = mysql.connector.connect(
        # host="192.168.0.102",
        host=DB_HOST,
        user="root",
        password="123456",
        database="ayesha",
        auth_plugin='mysql_native_password'
        )

    item_id_answer_map = {}
    item_id_difference_map = {}
    cursor = db.cursor()
    cursor.execute("select item_id, llm_answer, true_answer from llm_answer;")
    results = cursor.fetchall()
    cursor.close()
    db.close()
    for i in range(len(results)):
        item_id = results[i][0]
        llm_answer_json_str = results[i][1]
        llm_answer_list = json.loads(llm_answer_json_str)
        true_answer_json_str = results[i][1]
        true_answer_list = json.loads(true_answer_json_str)
        if item_id not in item_id_answer_map:
            item_id_answer_map[item_id] = {}
        if 'true_answer' not in item_id_answer_map[item_id]:
            item_id_answer_map[item_id]['true_answer'] = set()
        if 'llm_answer' not in item_id_answer_map[item_id]:
            item_id_answer_map[item_id]['llm_answer'] = set()
        for llm_answer in llm_answer_list:
            item_id_answer_map[item_id]['llm_answer'].add(llm_answer)
        for true_answer in true_answer_list:
            item_id_answer_map[item_id]['true_answer'].add(true_answer)

    for item_id,v in item_id_answer_map.items():
        if item_id not in item_id_difference_map:
            item_id_difference_map[item_id] = 0.0
        if len(v['true_answer']) == 0:
            continue
        item_id_difference_map[item_id] = float(len(v['llm_answer'])) / float(len(v['true_answer']))
    print(item_id_difference_map)
