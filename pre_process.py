from util.db_util import open_database
import json
from bs4 import BeautifulSoup
import jieba
import os
import itertools
import codecs


# DATA_FOLDER = "C:\hint-data"
DATA_FOLDER = "/Users/xiangchaolei/len/hint-data"
BATCH_SIZE = 100
MAX_INDEX = 100000

def find_max_index():
    maxDirIndex = 0
    indexFilePath = DATA_FOLDER + os.sep + "index.txt"
    if os.path.exists(indexFilePath):
        try:
            with open(DATA_FOLDER + os.sep + "index.txt", 'r') as f:
                maxDirIndex = int(f.readline())
        except:
            pass
    print("max dir index = ", maxDirIndex)
    return maxDirIndex

# 每10万个记录形成一个文件夹
# levelOne/levelTwo/levelThree/dividend
def index_to_path(itemIndex: int):
    levelOne = 0
    levelTwo = 0
    levelThree = 0
    dividend = itemIndex % 100;
    itemIndex //= 100
    levelThree = itemIndex % 100;
    itemIndex //= 100
    levelTwo = itemIndex % 100;
    itemIndex //= 100
    levelOne = itemIndex % 100;
    itemIndex //= 100
    return str(levelOne) + os.sep + str(levelTwo) + os.sep + str(levelThree) + os.sep + str(dividend)
    
def list_to_file(path: str, file_name: str, content: list):
    with open(path + os.sep + file_name, 'w', encoding="utf-8") as f:
        for s in content:
            f.write(s + '\n')

def main():
    db = open_database();
    maxIndex = find_max_index();
    while(maxIndex < MAX_INDEX):
        print("maxIndex = ", maxIndex)
        cursor = db.cursor()
        cursor.execute("SELECT * FROM item_doc where id > %(maxIndex)s order by id asc limit %(batchSize)s" , {"maxIndex": maxIndex, "batchSize": BATCH_SIZE})
        results = cursor.fetchall()
        print("1", len(results))
        for result in results:
            item = json.loads(result[1])
            maxIndex = int(result[4])
            print("maxIndex = ", maxIndex)
            # 提取信息
            contentHtml = item['bundler']['content']
            answerHtml = item['bundler']['answers']
            hintHtml = item['bundler']['hint']
            points = item["points"]
            # 去除html
            contentClean = BeautifulSoup(contentHtml, "html.parser").text
            answerClean = BeautifulSoup(answerHtml, "html.parser").text
            hintClean = BeautifulSoup(hintHtml, "html.parser").text
            # 分词
            contentSplit = list(jieba.cut(contentClean))
            answerSplit = list(jieba.cut(answerClean))
            hintSplit = list(jieba.cut(hintClean))
            # 去除相邻的重复元素
            contentSplit = [k for k, g in itertools.groupby(contentSplit)]
            answerSplit = [k for k, g in itertools.groupby(answerSplit)]
            hintSplit = [k for k, g in itertools.groupby(hintSplit)]
            pointList = []
            if points is not None:
                pointList = [val["name"] for i, val in enumerate(points)]
                pointList = [k for k, g in itertools.groupby(pointList)]
            # 路径
            path = DATA_FOLDER + os.sep + index_to_path(maxIndex)
            if not os.path.exists(path=path):
                os.makedirs(path)
            # 写入文件
            list_to_file(path, "answer.txt", answerSplit)
            list_to_file(path, "content.txt", contentSplit)
            list_to_file(path, "hint.txt", hintSplit)
            list_to_file(path, "point.txt", pointList)


            # with open(the_filename, 'r') as f:
            #     my_list = [line.rstrip('\n') for line in f]
        with open(DATA_FOLDER + os.sep + "index.txt", 'w') as f:
            f.write(str(maxIndex))
        cursor.close()
    db.close()

if __name__ == '__main__':
    # for i in range(2000000):
    #     path = DATA_FOLDER + os.sep + indexToPath(i)
    #     if not os.path.exists(path=path):
    #         os.makedirs(path)
    # main()
    db = open_database();
    cursor = db.cursor()
    cursor.execute("SELECT * FROM item_doc where id = 861785 order by id asc")
    results = cursor.fetchall()
    for result in results:
        print(result[1])
        print(result[2])
        print(type(result[2]))
        bytes_value = codecs.decode(result[2], 'hex_codec')
        print(bytes_value)
