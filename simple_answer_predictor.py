import torch
from simple_answer_model import Seq2SeqTransformer



# Model class must be defined somewhere
model = torch.load("./model")
model.eval()

