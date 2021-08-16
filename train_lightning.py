from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from bert.dataset import BertDataModule
from bert.model import BertModel
import argparse

from bidaf.dataset import BidafDataModule
from bidaf.model import BiDAF
from models import BertModelWithAdapters


def main(args):
    train_file = args.data_root + '/training_set_rel3.tsv'
    prompts_file = args.data_root + '/essay_prompts.json'
    if args.model == 'original':
        data = BertDataModule(
            train_file=train_file,
            prompts_file=prompts_file,
            bert_model=args.bert_model,
            batch_size=args.batch_size,
            seq_len=args.max_seq_length
        )
        model = BertModelWithAdapters(
            model_checkpoint=args.bert_model,
            lr=args.lr,
            prompts_file=args.prompts
        )
    elif args.model == 'bert':
        data = BertDataModule(
            train_file=train_file,
            prompts_file=prompts_file,
            bert_model=args.bert_model,
            batch_size=args.batch_size,
            seq_len=args.max_seq_length
        )
        model = BertModel(
            checkpoint=args.bert_model,
            prompts_file=args.prompts,
            lr=args.lr
        )
    else:
        data = BidafDataModule(
            train_file=train_file,
            prompts_file=prompts_file,
            batch_size=args.batch_size,
            seq_len=args.max_seq_length
        )
        model = BiDAF(
            hidden_size=args.hidden_size,
            seq_len=args.max_seq_length,
            lr=args.lr,
            drop_prob=0.2
        )
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[ModelCheckpoint(monitor="essay_set_avg")]
    )
    trainer.tune(
        model,
        datamodule=data
    )
    trainer.fit(
        model,
        datamodule=data
    )
    trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size per GPU. Scales automatically when \
                                  multiple GPUs are available.')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=100,
                        help='Number of features in encoder hidden layers.')
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=1024,
                        help='The maximum sequence length that is provided to the model.')
    parser.add_argument('--bert_model',
                        type=str,
                        default='bert-base-uncased',
                        choices=('bert-base-uncased', 'bert-large-uncased'),
                        help='The type of BERT model used.')
    parser.add_argument('--model',
                        type=str,
                        choices=('bert', 'bidaf', 'han', 'ensemble', 'original'),
                        help='The type of model used.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.00001,
                        help='Learning rate.')
    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')
    parser.add_argument('--data_root',
                        type=str,
                        default='./data',
                        help='Root file for data')
    parser.add_argument('--train_file',
                        type=str,
                        default='./data/training_set_rel3.tsv',
                        help='The CSV file to train')
    parser.add_argument('--prompts',
                        type=str,
                        default='./data/essay_prompts.json',
                        help='The JSON files with prompts')
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    seed_everything(args.seed, workers=True)

    main(args)
