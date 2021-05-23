import numpy as np

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizerFast


class BertDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, max_seq_length: int, doc_stride: int, prompts: dict, bert_model: str = 'bert-base-uncased'):
        self.datalist = dataset.drop(['domain1_score', 'domain2_score'], axis=1)
        self.domain1_scores = dataset['domain1_score'].tolist()
        self.domain1_scores_raw = dataset['domain1_score'].tolist()
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model)
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.prompts = prompts

        self.essay_ids = self.datalist['essay_id'].tolist()
        self.essay_sets = self.datalist['essay_set'].tolist()
        self.essays = self.datalist['essay'].tolist()

        self.x_encoded_bert = self.tokenizer(
            self.datalist['essay'].tolist(),
            max_length=max_seq_length,
            truncation=True,
            padding='max_length',
            return_overflowing_tokens=False,
            stride=doc_stride,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_offsets_mapping=True
        )

        self.domain1_scores = [(score - prompts[str(self.essay_sets[i])]['scoring']['domain1_score']['min_score']) / (
                    prompts[str(self.essay_sets[i])]['scoring']['domain1_score']['max_score'] -
                    prompts[str(self.essay_sets[i])]['scoring']['domain1_score']['min_score']) for i, score in
                               enumerate(self.domain1_scores)]

    def __getitem__(self, index: int):
        x = self.x_encoded_bert['input_ids'][index]
        mask = self.x_encoded_bert['attention_mask'][index]
        overflow_mapping = index
        essay_id = self.essay_ids[overflow_mapping]
        essay_set = self.essay_sets[overflow_mapping]
        prompt = self.prompts.get(str(essay_set))

        domain1_min_score = prompt['scoring']['domain1_score']['min_score']
        domain1_max_score = prompt['scoring']['domain1_score']['max_score']

        score1 = self.domain1_scores[overflow_mapping]

        dataitem = (
            essay_id,
            essay_set,
            x,
            mask,
            score1,
            domain1_min_score,
            domain1_max_score
        )

        return dataitem

    def __len__(self) -> int:
        return len(self.x_encoded_bert['input_ids'])


def collate_fn(examples):
    essay_ids = []
    essay_sets = []
    essays_bert = []
    scores_domain1 = []
    min_scores_domain1 = []
    max_scores_domain1 = []
    masks = []

    for essay_id, essay_set, bert, mask, score1, domain1_min_score, domain1_max_score in examples:
        essay_ids.append(essay_id)
        essay_sets.append(essay_set)
        masks.append(mask)
        essays_bert.append(bert)
        scores_domain1.append(score1)
        min_scores_domain1.append(domain1_min_score)
        max_scores_domain1.append(domain1_max_score)

    essay_ids = torch.LongTensor(essay_ids)
    essay_sets = torch.tensor(essay_sets, dtype=torch.long)
    essays_bert = torch.tensor(essays_bert, dtype=torch.long)
    masks = torch.tensor(masks, dtype=torch.bool)
    scores_domain1 = torch.tensor(scores_domain1, dtype=torch.float)
    min_scores_domain1 = torch.tensor(min_scores_domain1, dtype=torch.long)
    max_scores_domain1 = torch.tensor(max_scores_domain1, dtype=torch.long)

    return {
        'essay_ids': essay_ids,
        'essay_sets': essay_sets,
        'x': essays_bert,
        'masks': masks,
        'scores': scores_domain1,
        'min_scores': min_scores_domain1,
        'max_scores': max_scores_domain1
    }
