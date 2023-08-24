from torch.utils.data import IterableDataset 
import random
import mysql.connector
import json
from bs4 import BeautifulSoup
from typing import Iterable, List
from torchtext.vocab import build_vocab_from_iterator
from pre_process import indexToPath, DATA_FOLDER
import os

TRAIN_START_INDEX = 1
TRAIN_END_INDEX = 80

EVAL_START_INDEX = 81
EVAL_END_INDEX = 90

def openDatabase():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="123456",
        database="ayesha"
        )

INDEX_ID_MAP = {}

def getIndexIdMapping():
    db = openDatabase();
    cursor = db.cursor()
    cursor.execute("select id from item_processed where subject = 2 and department = 3 and length(knowledge) > 2;")
    results = cursor.fetchall()
    for i in range(len(results)):
        INDEX_ID_MAP[i+1]=results[i][0]
    cursor.close();
    db.close();

getIndexIdMapping()
# print(INDEX_ID_MAP)

def _read_text_iterator(startIndex: int, endIndex: int, category: str):
    # for index in range(startIndex, endIndex):
    #     filePath = DATA_FOLDER + os.sep + indexToPath(index) + os.sep + category + ".txt"
    #     yield readFileContent(filePath)
    for index in range(startIndex, endIndex):
        yield readFileContent(index, category)

def readFileContent(index: int, category: str):
    id = INDEX_ID_MAP[index]
    list = []
    # with open(path, 'r') as f:
    #     list = [line.rstrip('\n') for line in f]
    # return list
    db = openDatabase();
    cursor = db.cursor();
    if category == "input":
        cursor.execute("select content_vector, hint_vector from item_processed where id = %(id)s", { "id": id})
        result = cursor.fetchall()[0]
        contentVector = json.loads(result[0])
        for content in contentVector:
            list.append(content)
        list.append('<sep>')
        hintVector = json.loads(result[1])
        for hint in hintVector:
            list.append(hint)
    else:
        cursor.execute("select answer_vector from item_processed where id = %(id)s", { "id": id})
        result = cursor.fetchall()[0]
        answerVector = json.loads(result[0])
        for answer in answerVector:
            list.append(answer)
    cursor.close();
    db.close();
    return list;


class HintDataSet(IterableDataset): 
    def __init__(self, startIndex: int, endIndex: int):
        self.startIndex = startIndex
        self.endIndex = endIndex
        src_data_iter = _read_text_iterator(startIndex, endIndex, "input")
        trg_data_iter = _read_text_iterator(startIndex, endIndex, "output")
        self._iterator = zip(src_data_iter, trg_data_iter)
        self.current_pos = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_pos == (self.endIndex - self.startIndex + 1) - 1:
            raise StopIteration
        item = next(self._iterator)
        if self.current_pos is None:
            self.current_pos = 0
        else:
            self.current_pos += 1
        return item

    def __len__(self):
        return self.endIndex - self.startIndex + 1

    def pos(self):
        """
        Returns current position of the iterator. This returns None
        if the iterator hasn't been used yet.
        """
        return self.current_pos

    def __str__(self):
        return self.description
  

SRC = 'content_answer'
TGT = 'hint'

# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, side: str) -> List[str]:
    side_index = {SRC: 0, TGT: 1}
    for data_sample in data_iter:
        # print(list(jieba.cut(data_sample[side_index[side]])))
        yield data_sample[side_index[side]]

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX = 0, 1, 2, 3, 4
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>', '<sep>']

train_iter = HintDataSet(startIndex=TRAIN_START_INDEX, endIndex=EVAL_END_INDEX)
SRC_VOCAB_SIZE = len(build_vocab_from_iterator(yield_tokens(train_iter, SRC),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True))

train_iter = HintDataSet(startIndex=TRAIN_START_INDEX, endIndex=EVAL_END_INDEX)
TGT_VOCAB_SIZE = len(build_vocab_from_iterator(yield_tokens(train_iter, TGT),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True))

print("SRC_VOCAB_SIZE = ", SRC_VOCAB_SIZE)
print("TGT_VOCAB_SIZE = ", TGT_VOCAB_SIZE)


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


######################################################################
# During training, we need a subsequent word mask that will prevent the model from looking into
# the future words when making predictions. We will also need masks to hide
# source and target padding tokens. Below, let's define a function that will take care of both.
#


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


######################################################################
# Let's now define the parameters of our model and instantiate the same. Below, we also
# define our loss function which is the cross-entropy loss and the optimizer used for training.
#
torch.manual_seed(0)

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 32
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

######################################################################
# Collation
# ---------
#
# As seen in the ``Data Sourcing and Processing`` section, our data iterator yields a pair of raw strings.
# We need to convert these string pairs into the batched tensors that can be processed by our ``Seq2Seq`` network
# defined previously. Below we define our collate function that converts a batch of raw strings into batch tensors that
# can be fed directly into our model.
#


from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            # print(txt_input)
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
                                                    specials=special_symbols,
                                                    special_first=True)


train_iter = HintDataSet(startIndex=TRAIN_START_INDEX, endIndex=EVAL_END_INDEX)
vocab_transform[TGT] = build_vocab_from_iterator(yield_tokens(train_iter, TGT),
                                                    min_freq=1,
                                                    specials=special_symbols,
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

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

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

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

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
    torch.save(transformer, "./model")


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

