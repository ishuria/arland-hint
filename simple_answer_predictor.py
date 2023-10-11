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
    cursor.execute("select id from item_index_16 where item_type = 1 and is_chinese = 1 and `ignore` = 0;")
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

# train_iter = build_data_set(start_index=TRAIN_START_INDEX, end_index=EVAL_END_INDEX)
# vocab_transform[SRC] = build_vocab_from_iterator(yield_tokens(train_iter, SRC),
#                                                  min_freq=1,
#                                                  specials=SPECIAL_SYMBOLS,
#                                                  special_first=True)
#
# train_iter = build_data_set(start_index=TRAIN_START_INDEX, end_index=EVAL_END_INDEX)
# vocab_transform[TGT] = build_vocab_from_iterator(yield_tokens(train_iter, TGT),
#                                                  min_freq=1,
#                                                  specials=SPECIAL_SYMBOLS,
#                                                  special_first=True)
print("finished build_vocab_from_iterator")


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


# for side in [SRC, TGT]:
#     text_transform[side] = sequential_transforms(vocab_transform[side],  # Numericalization
#                                                  tensor_transform)  # Add BOS/EOS and create tensor


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


# for index in INDEX_ID_MAP:
#     if index <= EVAL_START_INDEX:
#         continue
#     src_sentence = read_file_content(INDEX_ID_MAP[index], ['content'])
#     answer_sentence = read_file_content_as_string(INDEX_ID_MAP[index], ['answer'])
#     src = text_transform[SRC](src_sentence).view(-1, 1)
#     num_tokens = src.shape[0]
#     src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
#     tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
#     print("----------------------")
#     print("original question: ", read_file_content_as_string(INDEX_ID_MAP[index], ['content']))
#     print("original answer: ", answer_sentence)
#     print("predicted answer: ",
#           " ".join(vocab_transform[TGT].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace(
#               "<eos>", ""))

def list_to_file(file_name: str, content: list):
    with open(file_name, 'w', encoding="utf-8") as f:
        for s in content:
            f.write(s + '\n')


contents = []
for index in INDEX_ID_MAP:
    content_sentence = read_file_content_as_string(INDEX_ID_MAP[index], ['content'], True)
    hint_sentence = read_file_content_as_string(INDEX_ID_MAP[index], ['hint'], True)
    knowledge_sentence = read_file_content_as_string(INDEX_ID_MAP[index], ['knowledge', 'auto_knowledge'], True)
    answer_sentence = read_file_content_as_string(INDEX_ID_MAP[index], ['answer'], True)
    if '图' in content_sentence:
        continue
    if len(hint_sentence) == 0:
        continue
    if 'A' in answer_sentence and 'B' in answer_sentence and 'C' in answer_sentence and 'D' in answer_sentence:
        continue
    if '故' in hint_sentence:
        continue
    print(INDEX_ID_MAP[index])
    content_sentence = content_sentence.replace("\t", " ")
    print('answer question: ', content_sentence, '\nhint: ', hint_sentence, '\nknowledge: ', knowledge_sentence)
    print('answer: ', answer_sentence)
    contents.append('\t'.join([str(INDEX_ID_MAP[index]), content_sentence, hint_sentence, knowledge_sentence, answer_sentence]))
list_to_file('/home/len/information-hint-data/kancd-data.txt', contents)

