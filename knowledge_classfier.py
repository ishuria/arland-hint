from util.db_util import open_database
from util.file_util import read_file_content

# 训练、验证参数
TRAIN_START_INDEX = 1
TRAIN_END_INDEX = 1500

EVAL_START_INDEX = 1501
EVAL_END_INDEX = 1950

# 序号 -> id
def get_index_id_mapping():
    index_id_map = {}
    db = open_database()
    cursor = db.cursor()
    cursor.execute("select id from item_index where subject = 2 and department = 3 and knowledge = 1;")
    results = cursor.fetchall()
    for i in range(len(results)):
        index_id_map[i + 1] = results[i][0]
    cursor.close()
    db.close()
    return index_id_map

INDEX_ID_MAP = get_index_id_mapping()

knowledge_map = {}
knowledge_index = 0

for i in range(len(INDEX_ID_MAP)):
    index = INDEX_ID_MAP[i+1]
    knowledge_list = read_file_content(index, ['knowledge'])
    for knowledge in knowledge_list:
        if knowledge not in knowledge_map:
            knowledge_map[knowledge] = knowledge_index
            knowledge_index+=1

for knowledge in knowledge_map:
    print(knowledge)