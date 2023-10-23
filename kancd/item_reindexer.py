import pandas as pd

train_data = pd.read_csv("/Users/xiangchaolei/len/train.csv")
test_data = pd.read_csv("/Users/xiangchaolei/len/test.csv")
df_item = pd.read_csv("/Users/xiangchaolei/len/item.csv")
item_id2new_item_id = {}
new_item_id = 1
for i, s in df_item.iterrows():
    item_id = s['item_id']
    if item_id not in item_id2new_item_id:
        item_id2new_item_id[item_id] = new_item_id
        new_item_id += 1
# print(item_id2new_item_id)

def convert_to_new_id(old_id):
    return item_id2new_item_id[old_id]

def convert_to_new_score(x):
    if x > 0:
        return 1.0
    else:
        return 0.0

# print(type(df_item))
df_item['item_id'] = df_item['item_id'].apply(convert_to_new_id)
df_item.to_csv('./kancd/data/new_item.csv', index=False)

train_data['item_id'] = train_data['item_id'].apply(convert_to_new_id)
train_data['score'] = train_data['score'].apply(convert_to_new_score)
train_data.to_csv('./kancd/data/new_train.csv', index=False)

test_data['item_id'] = test_data['item_id'].apply(convert_to_new_id)
test_data['score'] = test_data['score'].apply(convert_to_new_score)
test_data.to_csv('./kancd/data/new_test.csv', index=False)