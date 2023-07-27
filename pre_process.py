from torch.utils.data import IterableDataset 
import random
import mysql.connector
import json
from bs4 import BeautifulSoup
import jieba
from typing import Iterable, List
from torchtext.vocab import build_vocab_from_iterator

import os


DATA_FOLDER = "/Users/xiangchaolei/len/hint-data"
BATCH_SIZE = 100
MAX_INDEX = 100000

def findMaxIndex():
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
def indexToPath(itemIndex: int):
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


def openDatabase():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="123456",
        database="ayesha"
        )
    


def main():
    db = openDatabase();
    maxIndex = findMaxIndex();
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
            contentHtml = item['bundler']['content']
            answerHtml = item['bundler']['answers']
            hintHtml = item['bundler']['hint']
            contentClean = BeautifulSoup(contentHtml, "html.parser").text
            answerClean = BeautifulSoup(answerHtml, "html.parser").text
            hintClean = BeautifulSoup(hintHtml, "html.parser").text
            contentSplit = list(jieba.cut(contentClean))
            answerSplit = list(jieba.cut(answerClean))
            hintSplit = list(jieba.cut(hintClean))
            contentSplit.append('<sep>')
            inputList = contentSplit + answerSplit
            outputList = hintSplit
            path = DATA_FOLDER + os.sep + indexToPath(maxIndex)
            if not os.path.exists(path=path):
                os.makedirs(path)
            with open(path + os.sep + "input.txt", 'w') as f:
                for s in inputList:
                    f.write(s + '\n')
            with open(path + os.sep + "output.txt", 'w') as f:
                for s in outputList:
                    f.write(s + '\n')

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
    main()
