from torch.utils.data import IterableDataset
from typing import Iterable, List
from torchtext.vocab import build_vocab_from_iterator
from pre_process import indexToPath, DATA_FOLDER
import torch
import torch.nn as nn
from util.db_util import open_database
from util.file_util import read_file_content
from hint_data_set import HintDataSet
from transformer import Transformer, create_mask, generate_square_subsequent_mask

# 训练、验证参数
TRAIN_START_INDEX = 1
TRAIN_END_INDEX = 20000

EVAL_START_INDEX = 20001
EVAL_END_INDEX = 25000

# transformer参数
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


INDEX_ID_MAP = {}

SRC = 'input'
TGT = 'output'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_index_id_mapping():
    db = open_database();
    cursor = db.cursor()
    cursor.execute("select id from item_index where subject = 2 and department = 3;")
    results = cursor.fetchall()
    for i in range(len(results)):
        INDEX_ID_MAP[i+1]=results[i][0]
    cursor.close();
    db.close();

get_index_id_mapping()


def read_text_iterator(startIndex: int, endIndex: int, categories: list):
    for index in range(startIndex, endIndex):
        yield read_file_content(INDEX_ID_MAP[index], categories)

# def read_data_func(index: int, categories: list):
#     return read_file_content(INDEX_ID_MAP[index], categories)


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, side: str) -> List[str]:
    side_index = {SRC: 0, TGT: 1}
    for data_sample in data_iter:
        yield data_sample[side_index[side]]

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX = 0, 1, 2, 3, 4
# Make sure the tokens are in order of their indices to properly insert them in vocab
SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>', '<sep>']

def build_data_set(start_index: int, end_index: int):
    return HintDataSet(start_index=start_index, 
                       end_index=end_index,
                       read_text_iterator=read_text_iterator,
                       input_categories=['content', 'hint'],
                       output_categories=['answer'])


def count_vocabulary_size(train_iter: IterableDataset, side: str):
    return len(build_vocab_from_iterator(yield_tokens(train_iter, side),
                                         min_freq=1,
                                         specials=SPECIAL_SYMBOLS,
                                         special_first=True))
    
# 计算输入和输出字典的大小
SRC_VOCAB_SIZE = count_vocabulary_size(build_data_set(TRAIN_START_INDEX, EVAL_END_INDEX), SRC)
TGT_VOCAB_SIZE = count_vocabulary_size(build_data_set(TRAIN_START_INDEX, EVAL_END_INDEX), TGT)

print("SRC_VOCAB_SIZE = ", SRC_VOCAB_SIZE)
print("TGT_VOCAB_SIZE = ", TGT_VOCAB_SIZE)

transformer = Transformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


vocab_transform = {}

train_iter = HintDataSet(startIndex=TRAIN_START_INDEX, endIndex=EVAL_END_INDEX)
vocab_transform[SRC] = build_vocab_from_iterator(yield_tokens(train_iter, SRC),
                                                    min_freq=1,
                                                    specials=SPECIAL_SYMBOLS,
                                                    special_first=True)


train_iter = HintDataSet(startIndex=TRAIN_START_INDEX, endIndex=EVAL_END_INDEX)
vocab_transform[TGT] = build_vocab_from_iterator(yield_tokens(train_iter, TGT),
                                                    min_freq=1,
                                                    specials=SPECIAL_SYMBOLS,
                                                    special_first=True)

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
token_transform = {}

for side in [SRC, TGT]:
    text_transform[side] = sequential_transforms(vocab_transform[side], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC](src_sample))
        tgt_batch.append(text_transform[TGT](tgt_sample))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


from torch.utils.data import DataLoader

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = HintDataSet(startIndex=TRAIN_START_INDEX, endIndex=TRAIN_END_INDEX)
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, DEVICE)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = HintDataSet(startIndex=EVAL_START_INDEX, endIndex=EVAL_END_INDEX)
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, DEVICE)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)

######################################################################
# Now we have all the ingredients to train our model. Let's do it!
#

from timeit import default_timer as timer
NUM_EPOCHS = 18

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    torch.save(transformer, "./answer_model.bin")


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
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


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


######################################################################
#

print(translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu ."))

