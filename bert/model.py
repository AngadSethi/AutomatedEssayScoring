import torch
import torch.nn as nn
import transformers


class BERT(nn.Module):
    def __init__(self, model='bert-base-uncased'):
        super(BERT, self).__init__()

        self.bert_model = transformers.BertModel.from_pretrained(model, output_hidden_states=True)
        self.bert_config = self.bert_model.config

        self.tokenizer = transformers.BertTokenizer.from_pretrained(model, return_tensors="pt")

        self.projection = nn.Linear(
            in_features=self.bert_config.hidden_size,
            out_features=1,
            bias=True
        )

        self.activation = nn.Sigmoid()

    def forward(self, essay_ids, essay_sets, x, masks, scores, min_scores, max_scores):
        outputs = self.bert_model(x, attention_mask=masks)

        outputs = self.activation(self.projection(outputs.pooler_output))

        outputs = torch.squeeze(outputs, dim=-1)

        return outputs
