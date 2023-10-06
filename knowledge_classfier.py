from util.db_util import open_database
from util.file_util import read_file_content, read_file_content_as_string
import json

# OUTPUT_FOLDER = "/home/len/knowledge-data/"
OUTPUT_FOLDER = "/home/len/information-hint-data/"

# 训练、验证参数
TRAIN_START_INDEX = 1
TRAIN_END_INDEX = 11000

EVAL_START_INDEX = 11001
EVAL_END_INDEX = 14000


# 序号 -> id
def get_index_id_mapping():
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


INDEX_ID_MAP = get_index_id_mapping()

knowledge_map = {}
knowledge_index = 0

for i in range(len(INDEX_ID_MAP)):
    index = INDEX_ID_MAP[i + 1]
    knowledge_list = read_file_content(index, ['knowledge'], True)
    for knowledge in knowledge_list:
        if knowledge == '<sep>':
            continue
        if knowledge not in knowledge_map:
            knowledge_map[knowledge] = knowledge_index
            knowledge_index += 1

knowledge_list = []
for knowledge in knowledge_map:
    knowledge_list.append(knowledge)

print("total knowledge: ", len(knowledge_map))

with open(OUTPUT_FOLDER + 'class.txt', 'w') as class_file:
    class_file.write('\n'.join(knowledge_list))

train_list = []
for i in range(TRAIN_START_INDEX, TRAIN_END_INDEX + 1):
    index = INDEX_ID_MAP[i + 1]
    content = read_file_content_as_string(index, ['content', 'answer'], True)
    content = content.replace('<sep>', '')
    knowledge_list = read_file_content(index, ['knowledge'], True)
    knowledge_id_list = []
    for knowledge in knowledge_list:
        if knowledge == '<sep>':
            continue
        knowledge_id_list.append(knowledge_map[knowledge])
    train_list.append(content + '\t' + str(knowledge_id_list[0]))

with open(OUTPUT_FOLDER + 'train.txt', 'w') as train_file:
    train_file.write('\n'.join(train_list))

eval_list = []
for i in range(EVAL_START_INDEX, EVAL_END_INDEX + 1):
    index = INDEX_ID_MAP[i + 1]
    content = read_file_content_as_string(index, ['content', 'answer'], True)
    content = content.replace('<sep>', '')
    knowledge_list = read_file_content(index, ['knowledge'], True)
    knowledge_id_list = []
    for knowledge in knowledge_list:
        if knowledge == '<sep>':
            continue
        knowledge_id_list.append(knowledge_map[knowledge])
    eval_list.append(content + '\t' + str(knowledge_id_list[0]))

with open(OUTPUT_FOLDER + 'eval.txt', 'w') as eval_file:
    eval_file.write('\n'.join(eval_list))
