import json

import torch
from typing import Iterable, List
from util.db_util import open_database
from hint_data_set import HintDataSet
from util.file_util import read_file_content, read_file_content_as_string
from torchtext.vocab import build_vocab_from_iterator
from transformer import Seq2SeqTransformer, create_mask, generate_square_subsequent_mask

# 训练、验证参数
TRAIN_START_INDEX = 1
TRAIN_END_INDEX = 20000

EVAL_START_INDEX = 20001
EVAL_END_INDEX = 25000


# 序号 -> id
def get_index_id_mapping():
    index_id_map = {}
    db = open_database()
    cursor = db.cursor()
    cursor.execute("select id from item_index where subject = 4 and department = 3;")
    results = cursor.fetchall()
    for i in range(len(results)):
        index_id_map[i + 1] = results[i][0]
    cursor.close()
    db.close()
    return index_id_map


INDEX_ID_MAP = get_index_id_mapping()

# Model class must be defined somewhere
model = torch.load("./answer_model.bin")
model.eval()


def read_text_iterator(startIndex: int, endIndex: int, categories: list):
    for index in range(startIndex, endIndex):
        yield read_file_content(INDEX_ID_MAP[index], categories)


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def build_data_set(start_index: int, end_index: int):
    return HintDataSet(start_index=start_index,
                       end_index=end_index,
                       read_text_iterator=read_text_iterator,
                       input_categories=['content', 'hint'],
                       output_categories=['answer'])


text_transform = {}
token_transform = {}

SRC = 'input'
TGT = 'output'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, side: str) -> List[str]:
    side_index = {SRC: 0, TGT: 1}
    for data_sample in data_iter:
        yield data_sample[side_index[side]]


# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX = 0, 1, 2, 3, 4
# Make sure the tokens are in order of their indices to properly insert them in vocab
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>', '<sep>']

print("start build_vocab_from_iterator")
vocab_transform = {}

train_iter = build_data_set(start_index=TRAIN_START_INDEX, end_index=EVAL_END_INDEX)
vocab_transform[SRC] = build_vocab_from_iterator(yield_tokens(train_iter, SRC),
                                                 min_freq=1,
                                                 specials=SPECIAL_SYMBOLS,
                                                 special_first=True)

train_iter = build_data_set(start_index=TRAIN_START_INDEX, end_index=EVAL_END_INDEX)
vocab_transform[TGT] = build_vocab_from_iterator(yield_tokens(train_iter, TGT),
                                                 min_freq=1,
                                                 specials=SPECIAL_SYMBOLS,
                                                 special_first=True)
print("finished build_vocab_from_iterator")


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


for side in [SRC, TGT]:
    text_transform[side] = sequential_transforms(vocab_transform[side],  # Numericalization
                                                 tensor_transform)  # Add BOS/EOS and create tensor


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), DEVICE)
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


OUTPUT_FOLDER = "/home/len/knowledge-data/"
# 加载全部知识点
knowledge_list = []
knowledge_index_map = {}
with open(OUTPUT_FOLDER + 'class.txt', 'r') as f:
    knowledge_list = [line.rstrip('\n') for line in f]

for i in range(len(knowledge_list)):
    knowledge_index_map[knowledge_list[i]] = i + 1


item_file = ['item_id,knowledge_code']
train_file = ['user_id,item_id,score']
test_file = ['user_id,item_id,score']

for index in INDEX_ID_MAP:
    if index <= EVAL_START_INDEX:
        continue
    src_sentence = read_file_content(INDEX_ID_MAP[index], ['content', 'hint'])
    answer_sentence = read_file_content(INDEX_ID_MAP[index], ['answer'])

    item_knowledge_name_list = read_file_content(INDEX_ID_MAP[index], ['knowledge'])
    item_knowledge_id_list = []
    for item_knowledge_name in item_knowledge_name_list:
        if item_knowledge_name == '<sep>':
            continue
        item_knowledge_id_list.append(knowledge_index_map[item_knowledge_name])
    item_file.append(str(INDEX_ID_MAP[index]) + ',' + json.dumps(item_knowledge_id_list))


    src = text_transform[SRC](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    print("----------------------")
    print("original question: ", src_sentence)
    print("original answer: ", answer_sentence)
    pred_answer = " ".join(vocab_transform[TGT].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace(
              "<eos>", "")
    print("predicted answer: ", pred_answer)

    # print(torch.tensor(text_transform[TGT](answer_sentence), dtype=torch.int32).dtype)
    # print(tgt_tokens.dtype)
    loss = loss_fn(torch.tensor(text_transform[TGT](answer_sentence), dtype=torch.float32), torch.tensor(tgt_tokens.cpu(), dtype=torch.float32))
    print(loss)
    if index <= EVAL_START_INDEX + 0.7 * (EVAL_END_INDEX - EVAL_START_INDEX):
        train_file.append('1,' + str(INDEX_ID_MAP[index]) + ',' + str(loss))
    else:
        test_file.append('1,' + str(INDEX_ID_MAP[index]) + ',' + str(loss))


print(item_file)
print(train_file)
print(test_file)