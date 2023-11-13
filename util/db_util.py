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
    cursor.execute("select id from item_index_16 where is_chinese = 1 and `ignore` = 0 and knowledge = 1;")
    results = cursor.fetchall()
    for i in range(len(results)):
        index_id_map[i + 1] = results[i][0]
    cursor.close()
    db.close()
    return index_id_map