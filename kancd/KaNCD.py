# coding: utf-8
# 2023/3/7 @ WangFei
import logging
from EduCDM import KaNCD
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
torch.set_printoptions(threshold=100000)

df = pd.read_csv("./kancd/data/new_train.csv")
train_data, valid_data = train_test_split(df, test_size=0.2)
test_data = valid_data.copy()

train_data = pd.DataFrame(train_data, columns = df.columns).reset_index()
valid_data = pd.DataFrame(valid_data, columns = df.columns).reset_index()
test_data = pd.DataFrame(test_data, columns = df.columns).reset_index()

print("train_data = \n", train_data)
print("valid_data = \n", valid_data)
print("test_data = \n", test_data)

df_item = pd.read_csv("./kancd/data/new_item.csv")
item2knowledge = {}
knowledge_set = set()
for i, s in df_item.iterrows():
    item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
    item2knowledge[item_id] = knowledge_codes
    knowledge_set.update(knowledge_codes)

batch_size = 32
user_n = np.max(train_data['user_id'])
item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])])
knowledge_n = 524 #np.max(list(knowledge_set))


def transform(user, item, item2knowledge, score, batch_size):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
        torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


train_set, valid_set, test_set = [
    transform(data["user_id"], data["item_id"], item2knowledge, data["score"], batch_size)
    for data in [train_data, valid_data, test_data]
]

logging.getLogger().setLevel(logging.INFO)
cdm = KaNCD(exer_n=item_n, student_n=user_n, knowledge_n=knowledge_n, mf_type='gmf', dim=20)
cdm.train(train_set, valid_set, epoch_n=3, device="cpu", lr=0.002)
cdm.save("kancd.snapshot")

cdm.load("kancd.snapshot")
auc, accuracy = cdm.eval(test_set, device="cpu")
print("auc: %.6f, accuracy: %.6f" % (auc, accuracy))


