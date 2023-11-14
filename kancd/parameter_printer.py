from EduCDM import KaNCD
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
torch.set_printoptions(threshold=10000)

df = pd.read_csv("./data/new_train.csv")
train_data, valid_data = train_test_split(df, test_size=0.2)
test_data = valid_data.copy()

train_data = pd.DataFrame(train_data, columns = df.columns).reset_index()
valid_data = pd.DataFrame(valid_data, columns = df.columns).reset_index()
test_data = pd.DataFrame(test_data, columns = df.columns).reset_index()

print("train_data = \n", train_data)
print("valid_data = \n", valid_data)
print("test_data = \n", test_data)

df_item = pd.read_csv("./data/new_item.csv")
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


cdm = KaNCD(exer_n=item_n, student_n=user_n, knowledge_n=knowledge_n, mf_type='gmf', dim=20)
cdm.load("kancd.snapshot")

for name, param in cdm.net.named_parameters():
    if param.requires_grad:
        # print(name, param.data)
        print(name)
        # if name == 'prednet_full1.weight':
        #     print(name, param.data)