"""Obsolete Now. Datasets are now available in modules corresponding to the type of model.

Author:
    Angad Sethi
"""
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchtext
from nltk.tokenize import sent_tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class EssayDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, max_seq_length: int, max_query_length: int, doc_stride: int,
                 prompts: dict, total_seq_length: int, max_doc_length: int, max_sent_length: int, model: str = 'bert',
                 bert_model: str = 'bert-base-uncased', train: bool = True):
        self.datalist = dataset.drop(['domain1_score', 'domain2_score'], axis=1)
        self.domain1_scores = dataset['domain1_score'].tolist()
        self.domain1_scores_raw = dataset['domain1_score'].tolist()
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model)
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.prompts = prompts

        self.essay_ids = self.datalist['essay_id'].tolist()
        self.essay_sets = self.datalist['essay_set'].tolist()
        self.essays = self.datalist['essay'].tolist()

        self.vocab = torchtext.vocab.GloVe()
        self.en_tokenizer = torchtext.data.get_tokenizer('spacy')
        self.total_seq_length = total_seq_length

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

        self.model = model
        self.max_sent_length = max_sent_length
        self.max_doc_length = max_doc_length

        self.x_encoded_bidaf_list = []

        self.prompt_encoded_bidaf_list = []

        s = 'train_' if train else 'dev_'

        save_prompts = os.path.join('data', s + 'prompts_qlen_' + model + '_' + str(max_query_length) + '.pt')
        save_essays = os.path.join('data', s + 'essays_tlen_' + model + '_' + str(total_seq_length) + '.pt')

        if os.path.exists(save_prompts):
            self.prompt_encoded_bidaf_list = torch.load(save_prompts).tolist()
        else:
            for i in range(1, 9):
                prompt = self.prompts[str(i)]
                ans = [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['unk'] for token in
                       self.en_tokenizer(prompt['prompt'])]
                self.prompt_encoded_bidaf_list.append(ans)
            prompt_padded = pad_sequence(self.prompt_encoded_bidaf_list, batch_first=True)
            torch.save(prompt_padded, save_prompts)

        # if os.path.exists(save_essays):
        #     self.x_encoded_bidaf_list = torch.load(save_essays).tolist()
        # else:
        #     if model == 'han':
        #         for essay in tqdm(self.datalist['essay'], total=len(self.datalist['essay']), desc="Tokenizing Essays"):
        #             ans = [torch.LongTensor([self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['unk'] for token in
        #                    self.en_tokenizer(parts)[:max_sent_length]]) for parts in sent_tokenize(essay)[:max_doc_length]]
        #             self.x_encoded_bidaf_list.append(ans)
        #         essay_padded = pad_sequence(self.x_encoded_bidaf_list, batch_first=True)
        #         torch.save(essay_padded, save_essays)
        #     elif model == 'bidaf':
        #         for essay in tqdm(self.datalist['essay'], total=len(self.datalist['essay']), desc="Tokenizing Essays"):
        #             ans = torch.LongTensor([self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['unk'] for token in
        #                    self.en_tokenizer(essay)])
        #             self.x_encoded_bidaf_list.append(ans)
        #         essay_padded = pad_sequence(self.x_encoded_bidaf_list, batch_first=True)
        #         torch.save(essay_padded, save_essays)

        self.prompt_encoded_bidaf = self.prompt_encoded_bidaf_list
        # self.x_encoded_bidaf = self.x_encoded_bidaf_list

    def __getitem__(self, index: int) -> Tuple[int, int, list, list, list, list, np.int, int, int]:
        x = self.x_encoded_bert['input_ids'][index]
        mask = self.x_encoded_bert['attention_mask'][index]
        overflow_mapping = self.x_encoded_bert['overflow_to_sample_mapping'][index]
        essay_id = self.essay_ids[overflow_mapping]
        essay_set = self.essay_sets[overflow_mapping]
        prompt = self.prompts.get(str(essay_set))

        domain1_min_score = prompt['scoring']['domain1_score']['min_score']
        domain1_max_score = prompt['scoring']['domain1_score']['max_score']

        encoded_prompt = self.prompt_encoded_bidaf[int(essay_set) - 1]
        essay = self.essays[index]
        if self.model == 'han':
            encoded_x = [[self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['unk'] for token in
                          self.en_tokenizer(parts)[:self.max_sent_length]] for parts in
                         sent_tokenize(essay)[:self.max_doc_length]]
        else:
            encoded_x = [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['unk'] for token in
                         self.en_tokenizer(essay)[:self.total_seq_length]]

        score1 = self.domain1_scores[overflow_mapping]
        num_sents, num_words = 0, 0
        if self.model == 'han':
            num_sents = len(encoded_x)
            num_words = [len(x) for x in encoded_x]

        dataitem = (
            essay_id,
            essay_set,
            x,
            mask,
            encoded_x,
            encoded_prompt,
            score1,
            domain1_min_score,
            domain1_max_score,
            num_sents,
            num_words
        )

        return dataitem

    def __len__(self) -> int:
        return len(self.x_encoded_bert['input_ids'])


def collate_fn(examples):
    essay_ids = []
    essay_sets = []
    essays_bert = []
    essays_bidaf = []
    prompts = []
    scores_domain1 = []
    min_scores_domain1 = []
    max_scores_domain1 = []
    masks = []
    num_sents = []
    num_words = []

    for essay_id, essay_set, bert, mask, bidaf, prompt, score1, domain1_min_score, domain1_max_score, doc_length, sent_length in examples:
        essay_ids.append(essay_id)
        essay_sets.append(essay_set)
        masks.append(mask)
        essays_bert.append(bert)
        essays_bidaf.append(bidaf)
        prompts.append(prompt)
        scores_domain1.append(score1)
        min_scores_domain1.append(domain1_min_score)
        max_scores_domain1.append(domain1_max_score)
        num_sents.append(doc_length)
        num_words.append(sent_length)

    max_num_sents = max(num_sents)
    batch_size = len(essay_ids)
    max_num_words = max(max(x) for x in num_words)
    essay_ids = torch.LongTensor(essay_ids)
    essay_sets = torch.tensor(essay_sets, dtype=torch.long)
    essays_bert = torch.tensor(essays_bert, dtype=torch.long)
    masks = torch.tensor(masks, dtype=torch.bool)
    sentence_lengths = torch.zeros((batch_size,))
    if isinstance(essays_bidaf[0][0], list):
        essays_bidaf = torch.zeros((batch_size, max_num_sents, max_num_words))
        for doc_id, doc in enumerate(essays_bidaf):
            doc_length = num_sents[doc_id]
        essays_bidaf = torch.tensor(essays_bidaf, dtype=torch.long)
    else:
        essays_bidaf = torch.tensor(essays_bidaf, dtype=torch.long)
    prompts = torch.tensor(prompts, dtype=torch.long)
    scores_domain1 = torch.squeeze(torch.tensor(scores_domain1, dtype=torch.float), dim=-1)
    min_scores_domain1 = torch.tensor(min_scores_domain1, dtype=torch.long)
    max_scores_domain1 = torch.tensor(max_scores_domain1, dtype=torch.long)

    return essay_ids, essay_sets, essays_bert, masks, essays_bidaf, prompts, scores_domain1, min_scores_domain1, max_scores_domain1
