import mysql.connector
import json
from matplotlib import pyplot as plt
from util.config_util import DB_HOST
# import matplotlib
# matplotlib.rc("font",family='YouYuan')

if __name__ == "__main__":
    db = mysql.connector.connect(
        host="127.0.0.1",
        # host=DB_HOST,
        user="root",
        password="123456",
        database="ayesha",
        auth_plugin='mysql_native_password'
        )

    item_id_answer_map = {}
    item_id_difference_map = {}
    cursor = db.cursor()
    cursor.execute("select item_id, llm_answer, true_answer from llm_answer where llm_name like 'qwen%';")
    results = cursor.fetchall()
    cursor.close()
    db.close()
    for i in range(len(results)):
        item_id = results[i][0]
        llm_answer_json_str = results[i][1]
        llm_answer_list = json.loads(llm_answer_json_str)
        true_answer_json_str = results[i][2]
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
    item_id_difference_map = dict(sorted(item_id_difference_map.items(), key=lambda item: item[1]))

    unified_list_y = [0,0,0,0]
    unified_list_x = [1,2,3,4]
    for k,v in item_id_difference_map.items():
        if v == 4.0:
            unified_list_y[3]+=1
        if v == 3.0:
            unified_list_y[2]+=1
        if v == 2.0:
            unified_list_y[1]+=1
        if v == 1.0:
            unified_list_y[0]+=1

    plt.bar(unified_list_x, unified_list_y)
    plt.xlabel("试题不确定度")
    plt.ylabel("试题数量")

    plt.rcParams['font.sans-serif']=['Songti SC']
    plt.rcParams['axes.unicode_minus']=False
    # function to show the plot
    plt.show()

    db = mysql.connector.connect(
        # host="192.168.0.102",
        host=DB_HOST,
        user="root",
        password="123456",
        database="ayesha",
        auth_plugin='mysql_native_password'
        )
    cursor = db.cursor()
    for k,v in item_id_difference_map.items():
        if v >= 3.0:
            cursor.execute("""
        INSERT INTO `ayesha`.`difference_item_qwen`
        (`item_id`)
        VALUES
        (%(item_id)s);
        """ ,{
            "item_id": k
            })
    cursor.close()
    db.commit()
    db.close()
    print(item_id_difference_map)
