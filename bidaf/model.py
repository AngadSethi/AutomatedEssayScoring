import torch
import torch.nn as nn
import torchtext

import layers


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Borrowed from https://github.com/chrischute/squad

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        vocab (torchtext.vocab.GloVe): Pre-trained word vectors.
        seq_len (int): Maximum number of tokens allowed in the sequence.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, hidden_size: int, seq_len: int, drop_prob=0.):
        super(BiDAF, self).__init__()
        vocab = torchtext.vocab.GloVe()
        self.emb = layers.Embedding(vocab=vocab)

        self.enc = layers.RNNEncoder(input_size=vocab.dim,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size, seq_len=seq_len)

    def forward(self, essay_ids: torch.LongTensor, essay_sets: torch.LongTensor, x: torch.LongTensor,
                prompts: torch.LongTensor, scores: torch.FloatTensor, min_scores: torch.LongTensor,
                max_scores: torch.LongTensor) -> torch.Tensor:
        c_mask = torch.zeros_like(x) != x
        q_mask = torch.zeros_like(prompts) != prompts
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(x)  # (batch_size, c_len, hidden_size)
        q_emb = self.emb(prompts)  # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)  # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)  # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att[:, -1, :], mod[:, -1, :])

        return out
