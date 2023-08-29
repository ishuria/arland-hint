import os
from .config_util import DATA_FOLDER

# 每10万个记录形成一个文件夹
# levelOne/levelTwo/levelThree/dividend
def index_to_path(item_index: int):
    levelOne = 0
    levelTwo = 0
    levelThree = 0
    dividend = item_index % 100;
    item_index //= 100
    levelThree = item_index % 100;
    item_index //= 100
    levelTwo = item_index % 100;
    item_index //= 100
    levelOne = item_index % 100;
    item_index //= 100
    return str(levelOne) + os.sep + str(levelTwo) + os.sep + str(levelThree) + os.sep + str(dividend)
    
def read_file_content(item_index: int, categories: list):
    list = []
    for category in categories:
        with open(DATA_FOLDER + index_to_path(item_index) + os.sep + category + '.txt', 'r') as f:
            list = [line.rstrip('\n') for line in f]
        list.append('<sep>')
    return list
