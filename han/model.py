import torch
import torch.nn as nn

import layers


class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, word_hidden_size: int, sent_hidden_size: int, dropout: float = .2):
        super(HierarchicalAttentionNetwork, self).__init__()
        self.sent_hidden_size = sent_hidden_size
        self.word_hidden_size = word_hidden_size

        self.sentence_attention = layers.SentenceLevelAttention(sent_hidden_size, word_hidden_size, dropout)

        self.output = layers.HANOutput(2 * sent_hidden_size)

    def forward(self, essay_ids: torch.LongTensor, essay_sets: torch.LongTensor, x: torch.LongTensor,
                doc_lengths: torch.LongTensor, sentence_lengths: torch.LongTensor, scores: torch.FloatTensor,
                min_scores: torch.LongTensor, max_scores: torch.LongTensor) -> torch.Tensor:
        return self.output(self.sentence_attention(x, doc_lengths, sentence_lengths))
