"""Assortment of layers for use in models.py.

Author:
    Angad Sethi (angadsethi_2k18co066@dtu.ac.in)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.vocab
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Borrowed from https://github.com/chrischute/squad

    Args:
        vocab (torch.vocab.GloVe): Pre-trained word vectors.
    """

    def __init__(self, vocab: torchtext.vocab.GloVe):
        super(Embedding, self).__init__()
        self.vocab = vocab
        self.embedding = nn.Embedding.from_pretrained(self.vocab.vectors, freeze=True)

    def forward(self, x):
        emb = self.embedding(x)  # (batch_size, seq_len, embed_size)

        return emb


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Borrowed from https://github.com/chrischute/squad

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths: torch.Tensor):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        lengths = torch.as_tensor(lengths, dtype=torch.int64, device=torch.device('cpu'))
        x = x[sort_idx]  # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Borrowed from https://github.com/chrischute/squad

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)  # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)  # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)  # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2) \
            .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    @author: Chris Schute (https://github.com/chrischute/squad)

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, seq_len):
        super(BiDAFOutput, self).__init__()
        self.linear_1 = nn.Linear(1 * hidden_size, 1 * hidden_size)
        self.activation_1 = nn.Tanh()
        self.dropout_1 = nn.Dropout(0.1)
        self.linear_2 = nn.Linear(1 * hidden_size, 1)
        self.activation_2 = nn.Sigmoid()

    def forward(self, att, mod):
        # logits_1 = torch.cat([att, mod], -1)
        logits_1 = mod

        logits_1 = self.linear_1(logits_1)
        logits_1 = self.activation_1(logits_1)
        logits_1 = self.dropout_1(logits_1)

        logits_1 = self.linear_2(logits_1)
        logits_1 = self.activation_2(logits_1)

        logits_1 = torch.squeeze(logits_1, -1)

        return logits_1


class GRUEncoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_size: int):
        super(GRUEncoder, self).__init__()
        self.encoder = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, embeddings):
        return self.encoder(embeddings)


class WordLevelAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(WordLevelAttention, self).__init__()
        self.word_linear = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.activation = nn.Tanh()
        self.attention = nn.Linear(2 * hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.vocab = torchtext.vocab.GloVe()
        self.encoder = GRUEncoder(self.vocab.dim, hidden_size)
        self.hidden_size = hidden_size
        self.word_embedding = Embedding(self.vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences, sentence_lengths):
        sentence_lengths, sentence_indices = torch.sort(sentence_lengths, dim=0, descending=True)
        sentences = sentences[sentence_indices]

        sentences = self.word_embedding(sentences)
        sentences = self.dropout(sentences)

        packed_words = pack_padded_sequence(sentences, sentence_lengths.tolist(), batch_first=True)
        encoded_output, _ = self.encoder(packed_words)
        sentences, _ = pad_packed_sequence(encoded_output, batch_first=True)

        output = self.word_linear(sentences)
        output = self.activation(output)
        # Output -> (batch, seq_len, 2 * hidden_size)
        attention_weights = self.softmax(self.attention(output))
        # Output -> (batch, seq_len, 1)
        attention_weights = attention_weights.expand(-1, -1, 2 * self.hidden_size)
        output = attention_weights * sentences
        # Output -> (batch, seq_len, 2  * hidden_size)
        output = torch.sum(output, dim=1)
        # Output -> (batch, 2  * hidden_size)
        _, sentence_indices = torch.sort(sentence_indices, dim=0)
        output = output[sentence_indices]
        return output


class SentenceLevelAttention(nn.Module):
    def __init__(self, sent_hidden_size: int, word_hidden_size: int, dropout: float = 0.1):
        super(SentenceLevelAttention, self).__init__()
        self.sent_linear = nn.Linear(2 * sent_hidden_size, 2 * sent_hidden_size)
        self.activation = nn.Tanh()
        self.attention = nn.Linear(2 * sent_hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.word_attention = WordLevelAttention(word_hidden_size, dropout)
        self.sentence_encoder = GRUEncoder(2 * word_hidden_size, sent_hidden_size)

    def forward(self, docs, doc_lengths, sent_lengths):
        doc_lengths, doc_indices = torch.sort(doc_lengths, dim=0, descending=True)
        docs = docs[doc_indices]
        sent_lengths = sent_lengths[doc_indices]

        # 'docs' is a three-dimensional tensor. Each doc has sentences and each sentence has tokens.
        # By doing this, we are effectively packing all documents into a two-dimensional tensor of
        # shape (num_sentences, tokens)
        packed_docs = pack_padded_sequence(docs, doc_lengths.tolist(), batch_first=True)
        packed_doc_lengths = pack_padded_sequence(sent_lengths, doc_lengths.tolist(), batch_first=True)
        valid_batch = packed_docs.batch_sizes

        packed_docs = self.word_attention(packed_docs.data, packed_doc_lengths.data)

        packed_docs, _ = self.sentence_encoder(PackedSequence(packed_docs, valid_batch))
        encoded_output, _ = pad_packed_sequence(packed_docs, batch_first=True)

        output = self.sent_linear(encoded_output)
        output = self.activation(output)
        attention_weights = self.attention(output)
        attention_weights = self.softmax(attention_weights)
        output = attention_weights * encoded_output
        output = torch.sum(output, dim=1)
        _, doc_indices = torch.sort(doc_indices, dim=0)
        output = output[doc_indices]

        return output


class HANOutput(nn.Module):
    def __init__(self, hidden_size):
        super(HANOutput, self).__init__()
        self.projection = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, pred):
        pred = self.projection(pred)
        pred = self.activation(pred)
        pred = torch.squeeze(pred, -1)

        return pred
