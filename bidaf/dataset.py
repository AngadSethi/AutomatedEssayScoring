import os

import pandas as pd
import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class BiDAFDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, prompts: dict, seq_len: int, mode: str = 'train'):
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

        self.x_encoded_bidaf_list = []

        self.prompt_encoded_bidaf_list = []

        save_essays = os.path.join('data', mode + 'essays_tlen_' + str(seq_len) + '.pt')
        for i in range(1, 9):
            prompt = self.prompts[str(i)]
            ans = torch.LongTensor(
                [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['unk'] for token in
                 self.en_tokenizer(prompt['prompt'])])
            self.prompt_encoded_bidaf_list.append(ans)
        self.prompt_encoded_bidaf_list = pad_sequence(self.prompt_encoded_bidaf_list, batch_first=True)

        if os.path.exists(save_essays):
            self.x_encoded_bidaf_list = torch.load(save_essays)
        else:
            for essay in tqdm(self.datalist['essay'], total=len(self.datalist['essay']), desc="Tokenizing Essays"):
                ans = torch.zeros(seq_len, dtype=torch.long)
                tokens = self.en_tokenizer(essay)[:seq_len]
                ans[:len(tokens)] = torch.LongTensor(
                    [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi['unk'] for token in
                     tokens])
                self.x_encoded_bidaf_list.append(ans)
            self.x_encoded_bidaf_list = torch.stack(self.x_encoded_bidaf_list)
            torch.save(self.x_encoded_bidaf_list, save_essays)

        self.prompt_encoded_bidaf = self.prompt_encoded_bidaf_list.tolist()
        self.x_encoded_bidaf = self.x_encoded_bidaf_list.tolist()

    def __getitem__(self, index: int):
        essay_id = self.essay_ids[index]
        essay_set = self.essay_sets[index]
        prompt = self.prompts.get(str(essay_set))

        domain1_min_score = prompt['scoring']['domain1_score']['min_score']
        domain1_max_score = prompt['scoring']['domain1_score']['max_score']

        encoded_prompt = self.prompt_encoded_bidaf[int(essay_set) - 1]
        encoded_x = self.x_encoded_bidaf[index]

        score1 = self.domain1_scores[index]

        dataitem = (
            essay_id,
            essay_set,
            encoded_x,
            encoded_prompt,
            score1,
            domain1_min_score,
            domain1_max_score
        )

        return dataitem

    def __len__(self) -> int:
        return len(self.essay_ids)


def collate_fn(examples):
    essay_ids = []
    essay_sets = []
    essays_bidaf = []
    prompts = []
    scores_domain1 = []
    min_scores_domain1 = []
    max_scores_domain1 = []

    for essay_id, essay_set, bidaf, prompt, score1, domain1_min_score, domain1_max_score in examples:
        essay_ids.append(essay_id)
        essay_sets.append(essay_set)
        essays_bidaf.append(bidaf)
        prompts.append(prompt)
        scores_domain1.append(score1)
        min_scores_domain1.append(domain1_min_score)
        max_scores_domain1.append(domain1_max_score)

    essay_ids = torch.LongTensor(essay_ids)
    essay_sets = torch.tensor(essay_sets, dtype=torch.long)
    essays_bidaf = torch.tensor(essays_bidaf, dtype=torch.long)
    prompts = torch.tensor(prompts, dtype=torch.long)
    scores_domain1 = torch.tensor(scores_domain1, dtype=torch.float)
    min_scores_domain1 = torch.tensor(min_scores_domain1, dtype=torch.long)
    max_scores_domain1 = torch.tensor(max_scores_domain1, dtype=torch.long)

    return {
        'essay_ids': essay_ids,
        'essay_sets': essay_sets,
        'x': essays_bidaf,
        'prompts': prompts,
        'scores': scores_domain1,
        'min_scores': min_scores_domain1,
        'max_scores': max_scores_domain1
    }
