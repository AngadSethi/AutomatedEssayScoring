import torch.nn as nn
import layers


class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, dropout=.2):
        super(HierarchicalAttentionNetwork, self).__init__()
        self.sent_hidden_size = sent_hidden_size
        self.word_hidden_size = word_hidden_size

        self.sentence_attention = layers.SentenceLevelAttention(sent_hidden_size, word_hidden_size, dropout)

        self.output = layers.HANOutput(2 * sent_hidden_size)

    def forward(self, essay_ids, essay_sets, x, doc_lengths, sentence_lengths, scores, min_scores, max_scores):
        return self.output(self.sentence_attention(x, doc_lengths, sentence_lengths))
