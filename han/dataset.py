import torch
from torch.utils.data import Dataset
import pandas as pd
import torchtext
from nltk.tokenize import sent_tokenize


class HANDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, prompts: dict, max_doc_length: int, max_sent_length: int):
        self.datalist = dataset.drop(['domain1_score', 'domain2_score'], axis=1)
        self.domain1_scores = dataset['domain1_score'].tolist()
        self.domain1_scores_raw = dataset['domain1_score'].tolist()
        self.prompts = prompts

        self.essay_ids = self.datalist['essay_id'].tolist()
        self.essay_sets = self.datalist['essay_set'].tolist()
        self.essays = self.datalist['essay'].tolist()

        self.vocab = torchtext.vocab.GloVe()
        self.en_tokenizer = torchtext.data.get_tokenizer('spacy')

        self.domain1_scores = [(score - prompts[str(self.essay_sets[i])]['scoring']['domain1_score']['min_score']) / (
                    prompts[str(self.essay_sets[i])]['scoring']['domain1_score']['max_score'] -
                    prompts[str(self.essay_sets[i])]['scoring']['domain1_score']['min_score']) for i, score in
                               enumerate(self.domain1_scores)]

        self.max_sent_length = max_sent_length
        self.max_doc_length = max_doc_length

    def __getitem__(self, index: int):
        essay_id = self.essay_ids[index]
        essay_set = self.essay_sets[index]
        prompt = self.prompts.get(str(essay_set))

        domain1_min_score = prompt['scoring']['domain1_score']['min_score']
        domain1_max_score = prompt['scoring']['domain1_score']['max_score']

        essay = self.essays[index]
        encoded_x = [[self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['unk'] for token in
             self.en_tokenizer(parts)[:self.max_sent_length]] for parts in sent_tokenize(essay)[:self.max_doc_length]]

        score1 = self.domain1_scores[index]
        num_sents = len(encoded_x)
        num_words = [len(x) for x in encoded_x]

        dataitem = (
            essay_id,
            essay_set,
            encoded_x,
            score1,
            domain1_min_score,
            domain1_max_score,
            num_sents,
            num_words
        )

        return dataitem

    def __len__(self) -> int:
        return len(self.essay_ids)


def collate_fn(examples):
    essay_ids = []
    essay_sets = []
    essays_bidaf = []
    scores_domain1 = []
    min_scores_domain1 = []
    max_scores_domain1 = []
    num_sents = []
    num_words = []

    for essay_id, essay_set, bidaf, score1, domain1_min_score, domain1_max_score, doc_length, sent_length in examples:
        essay_ids.append(essay_id)
        essay_sets.append(essay_set)
        essays_bidaf.append(bidaf)
        scores_domain1.append(score1)
        min_scores_domain1.append(domain1_min_score)
        max_scores_domain1.append(domain1_max_score)
        num_sents.append(doc_length)
        num_words.append(sent_length)

    max_num_sents = max(num_sents)
    batch_size = len(essay_ids)
    max_num_words = max([max(x) for x in num_words])
    essay_ids = torch.LongTensor(essay_ids)
    essay_sets = torch.LongTensor(essay_sets)
    sentence_lengths = torch.zeros((batch_size, max_num_sents), dtype=torch.long)
    doc_lengths = torch.LongTensor(num_sents)
    essays_bidaf_final = torch.zeros((batch_size, max_num_sents, max_num_words), dtype=torch.long)
    for doc_id, doc in enumerate(essays_bidaf):
        num_sent = num_sents[doc_id]
        sentence_lengths[doc_id, :num_sent] = torch.LongTensor(num_words[doc_id])
        for i in range(num_sent):
            num_word = num_words[doc_id][i]
            essays_bidaf_final[doc_id, i, :num_word] = torch.LongTensor(doc[i])
    scores_domain1 = torch.FloatTensor(scores_domain1)
    min_scores_domain1 = torch.LongTensor(min_scores_domain1)
    max_scores_domain1 = torch.LongTensor(max_scores_domain1)

    return {
        'essay_ids': essay_ids,
        'essay_sets': essay_sets,
        'x': essays_bidaf_final,
        'doc_lengths': doc_lengths,
        'sentence_lengths': sentence_lengths,
        'scores': scores_domain1,
        'min_scores': min_scores_domain1,
        'max_scores': max_scores_domain1
    }
