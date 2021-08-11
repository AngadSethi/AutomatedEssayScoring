import torch
import torch.nn as nn
import torchtext
from torch import optim
import torch.nn.functional as F

import layers
from pytorch_lightning import LightningModule

from util import quadratic_weighted_kappa, log_final_results


class BiDAF(LightningModule):
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

    def __init__(self, hidden_size: int, seq_len: int, lr: float, drop_prob=0.):
        super().__init__()
        vocab = torchtext.vocab.GloVe()
        self.emb = layers.Embedding(vocab=vocab)
        self.lr = lr
        self.enc = layers.RNNEncoder(input_size=vocab.dim,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size, seq_len=seq_len)

    def configure_optimizers(self):
        optimizer = optim.SGD(params=filter(lambda x: x.requires_grad, self.parameters()), lr=self.lr, momentum=0.9)
        return optimizer

    def forward(self, x: torch.LongTensor, prompts: torch.LongTensor) -> torch.Tensor:
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

    def training_step(self, batch, batch_idx):
        essay_ids, essay_sets, x, prompts, scores, min_scores, max_scores = batch
        predictions = self(x, prompts)
        loss = F.mse_loss(predictions, scores)
        return loss

    def validation_step(self, batch, batch_idx):
        essay_ids, essay_sets, x, prompts, scores, min_scores, max_scores = batch
        predictions = self(x, prompts)
        loss = F.mse_loss(predictions, scores)

        scaled_predictions = min_scores + ((max_scores - min_scores) * predictions)
        scores_domain1 = min_scores + ((max_scores - min_scores) * scores)

        quadratic_kappa_overall = quadratic_weighted_kappa(
            torch.round(scaled_predictions).type(torch.IntTensor).tolist(),
            torch.round(scores_domain1).type(torch.IntTensor).tolist(),
            min_rating=0,
            max_rating=60
        )

        self.log_dict({'val_loss': loss, 'quadratic_kappa_overall_dev': quadratic_kappa_overall})

        return {
            'essay_ids': essay_ids,
            'essay_sets': essay_sets,
            'loss': loss,
            'predictions': scaled_predictions,
            'scores': scores_domain1
        }

    def test_step(self, batch, batch_idx):
        essay_ids, essay_sets, x, prompts, scores, min_scores, max_scores = batch
        predictions = self(x, prompts)
        loss = F.mse_loss(predictions, scores)

        scaled_predictions = min_scores + ((max_scores - min_scores) * predictions)
        scores_domain1 = min_scores + ((max_scores - min_scores) * scores)

        quadratic_kappa_overall = quadratic_weighted_kappa(
            torch.round(scaled_predictions).type(torch.IntTensor).tolist(),
            torch.round(scores_domain1).type(torch.IntTensor).tolist(),
            min_rating=0,
            max_rating=60
        )

        self.log_dict({'test_loss': loss, 'quadratic_kappa_overall_test': quadratic_kappa_overall})

        return {
            'essay_ids': essay_ids,
            'essay_sets': essay_sets,
            'loss': loss,
            'predictions': scaled_predictions,
            'scores': scores_domain1
        }

    def validation_epoch_end(self, outputs):
        print(f"Val Results: {dumps(log_final_results(outputs, self.prompts), indent=4)}")

    def test_epoch_end(self, outputs):
        print(f"Test Results: {dumps(log_final_results(outputs, self.prompts), indent=4)}")
