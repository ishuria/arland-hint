import torch.nn as nn
import torch

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        out = self.embed(x)
        return out
    
if __name__=="__main__":
    embedding = Embedding(8, 3)
    print(embedding.forward(torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])))