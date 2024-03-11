import mysql.connector
from .config_util import DB_HOST

def open_database():
    return mysql.connector.connect(
        # host="192.168.0.102",
        host=DB_HOST,
        user="root",
        password="123456",
        database="ayesha",
        auth_plugin='mysql_native_password'
        )

def get_it_item_index_id_mapping():
    index_id_map = {}
    db = open_database()
    cursor = db.cursor()
    cursor.execute("select id, item_id from item_index_16 where is_chinese = 1 and `ignore` = 0 and knowledge = 1;")
    results = cursor.fetchall()
    for i in range(len(results)):
        index_id_map[i + 1] = results[i][0]
    cursor.close()
    db.close()
    return index_id_map

def get_it_item_index_id_item_id_mapping():
    index_id_map = {}
    id_item_id_map = {}
    db = open_database()
    cursor = db.cursor()
    cursor.execute("select id, item_id from item_index_16 where is_chinese = 1 and `ignore` = 0 and knowledge = 1 and item_type = 1;")
    results = cursor.fetchall()
    for i in range(len(results)):
        index_id_map[i + 1] = results[i][0]
        id_item_id_map[results[i][0]] = results[i][1]
    cursor.close()
    db.close()
    return index_id_map, id_item_id_map

def save_llm_answer(llm_name: str, llm_original_answer: str, llm_answer: str, score: float, item_id: int):
    db = open_database()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO `ayesha`.`llm_answer`
        (`llm_name`,
        `llm_original_answer`,
        `llm_answer`,
        `score`,
        `item_id`)
        VALUES
        (%(llm_name)s,
        %(llm_original_answer)s,
        %(llm_answer)s,
        %(score)s, 
        %(item_id)s);
        """ ,{
            "llm_name": llm_name,
            "llm_original_answer": llm_original_answer,
            "llm_answer": llm_answer,
            "score": score,
            "item_id": item_id
            })
    cursor.close()
    db.commit()
    db.close()