import argparse
import logging

import numpy as np
import torch
from pytorch_lightning import Trainer

from dataset import ArgsBase, NSMCDataModule, KorSTSDataModule
from model import SubtaskGPT2, SubtaskGPT2Regression, Classification

parser = argparse.ArgumentParser(description='Train KoGPT2 subtask model')

parser.add_argument('--task', type=str, default=None, help='subtask name')

if __name__ == '__main__':
    parser = ArgsBase.add_model_specific_args(parser)
    parser = Classification.add_model_specific_args(parser)
    parser = NSMCDataModule.add_model_specific_args(parser)
    parser = KorSTSDataModule.add_model_specific_args(parser)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    
    if args.task.lower() == 'nsmc':
        dm = NSMCDataModule(args.train_data_path,
                            args.val_data_path,
                            batch_size=args.batch_size,
                            max_seq_len=args.seq_len,
                            num_workers=args.num_workers)
        args.num_labels = 2
        model = SubtaskGPT2(args)
    elif args.task.lower() == 'korsts':
        dm = KorSTSDataModule(args.train_data_path,
                              args.val_data_path,
                              batch_size=args.batch_size,
                              max_seq_len=args.seq_len,
                              num_workers=args.num_workers)
        args.num_labels = 1
        model = SubtaskGPT2Regression(args)
    else:
        assert False, 'no task matched!'

    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model, datamodule=dm)
