from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from bert.dataset import BertDataModule
from bert.model import BertModel
import argparse

from bidaf.dataset import BidafDataModule
from bidaf.model import BiDAF
from models import BertModelWithAdapters


def main(args):
    dict_args = vars(args)
    train_file = args.data_root + '/training_set_rel3.tsv'
    prompts_file = args.data_root + '/essay_prompts.json'
    if args.model == 'original':
        data = BertDataModule(
            train_file=train_file,
            prompts_file=prompts_file,
            bert_model=args.bert_model,
            batch_size=args.batch_size,
            seq_len=args.max_seq_length,
            essay_set=args.essay_set
        )
        model = BertModelWithAdapters(**dict_args)
    elif args.model == 'bert':
        data = BertDataModule(
            train_file=train_file,
            prompts_file=prompts_file,
            bert_model=args.bert_model,
            batch_size=args.batch_size,
            seq_len=args.max_seq_length,
            essay_set=args.essay_set
        )
        model = BertModel(**dict_args)
    else:
        data = BidafDataModule(
            train_file=train_file,
            prompts_file=prompts_file,
            batch_size=args.batch_size,
            seq_len=args.max_seq_length,
            essay_set=args.essay_set
        )
        model = BiDAF(**dict_args)
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[ModelCheckpoint(monitor="essay_set_avg")]
    )
    # trainer.tune(
    #     model,
    #     datamodule=data
    # )
    trainer.fit(
        model,
        datamodule=data
    )
    trainer.test()


if __name__ == '__main__':
    model = 'original'
    gpus = 1

    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--model',
                        type=str,
                        choices=('bert', 'bidaf', 'han', 'ensemble', 'original'),
                        help='The type of model used.')
    temp_args, _ = parser.parse_known_args()
    temp_args.model = model
    # let the model add what it wants
    if temp_args.model == "original":
        parser = BertModelWithAdapters.add_model_specific_args(parser)
    elif temp_args.model == "bert":
        parser = BertModel.add_model_specific_args(parser)
    elif temp_args.model == "bidaf":
        parser = BertModelWithAdapters.add_model_specific_args(parser)

    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')
    parser.add_argument('--essay_set',
                        type=int,
                        default=0,
                        help='Which essay set to train on. 0 means all')
    parser.add_argument('--data_root',
                        type=str,
                        default='./data',
                        help='Root file for data')
    parser.add_argument('--train_file',
                        type=str,
                        default='./data/training_set_rel3.tsv',
                        help='The CSV file to train')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size per GPU. Scales automatically when \
                                  multiple GPUs are available.')
    args, _ = parser.parse_known_args()
    args.model = model
    args.gpus = gpus
    seed_everything(args.seed, workers=True)

    main(args)
