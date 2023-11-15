import os
from .config_util import DATA_FOLDER
import json


# 每10万个记录形成一个文件夹
# levelOne/levelTwo/levelThree/dividend
def index_to_path(item_index: int):
    dividend = item_index % 100;
    item_index //= 100
    levelThree = item_index % 100;
    item_index //= 100
    levelTwo = item_index % 100;
    item_index //= 100
    levelOne = item_index % 100;
    item_index //= 100
    return str(levelOne) + os.sep + str(levelTwo) + os.sep + str(levelThree) + os.sep + str(dividend)


def read_file_content(item_index: int, categories: list, no_sep: bool):
    list = []
    for category in categories:
        _list = []
        if not os.path.isfile(DATA_FOLDER + index_to_path(item_index) + os.sep + category + '.txt'):
            continue
        with open(DATA_FOLDER + index_to_path(item_index) + os.sep + category + '.txt', 'r', encoding="utf-8") as f:
            _list = [line.rstrip('\n') for line in f]
        list.extend(_list)
        if not no_sep:
            list.append('<sep>')
    return list


def read_file_content_as_string(item_index: int, categories: list, no_sep: bool):
    list = read_file_content(item_index, categories, no_sep)
    return "".join(list)

def read_file_content_as_array(item_index: int, categories: list, no_sep: bool):
    str_content = read_file_content_as_string(item_index, categories, no_sep)
    return json.loads(str_content)