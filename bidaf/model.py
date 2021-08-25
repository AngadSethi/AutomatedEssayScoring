import torch
import torch.nn as nn
import torchtext
from torch import optim
import torch.nn.functional as F

import layers
from pytorch_lightning import LightningModule

from util import quadratic_weighted_kappa, log_final_results
from json import dumps


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

    def __init__(self, lr: float, hidden_size: int, max_seq_length: int, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        vocab = torchtext.vocab.GloVe()
        self.emb = layers.Embedding(vocab=vocab)
        self.enc = layers.RNNEncoder(input_size=vocab.dim,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=0.1)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=0.1)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=0.1)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size, seq_len=max_seq_length)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BidafModel")
        parser.add_argument('--hidden_size',
                            type=int,
                            default=100,
                            help='Number of features in encoder hidden layers.')
        parser.add_argument('--lr',
                            type=float,
                            default=0.00001,
                            help='Learning rate.')
        parser.add_argument('--max_seq_length',
                            type=int,
                            default=1024,
                            help='The maximum sequence length that is provided to the model.')
        return parent_parser

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

        self.log('val_loss', loss)

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

        self.log('test_loss', loss)

        return {
            'essay_ids': essay_ids,
            'essay_sets': essay_sets,
            'loss': loss,
            'predictions': scaled_predictions,
            'scores': scores_domain1
        }

    def validation_epoch_end(self, outputs):
        final_results = log_final_results(outputs, self.prompts)
        self.log_dict(final_results)

        print(f"Val Results: {dumps(final_results, indent=4)}")

    def test_epoch_end(self, outputs):
        final_results = log_final_results(outputs, self.prompts)
        self.log_dict(final_results)

        print(f"Test Results: {dumps(final_results, indent=4)}")
