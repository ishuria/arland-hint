import torch


# Model class must be defined somewhere
model = torch.load("./answer_model.bin")
model.eval()

