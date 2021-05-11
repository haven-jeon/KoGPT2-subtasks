import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import PreTrainedTokenizerFast


class NSMCDataset(Dataset):
    def __init__(self, datapath, max_seq_len=128):
        self.datapath = datapath
        self.data = pd.read_csv(self.datapath, sep='\t')
        self.bos_token = '</s>'
        self.eos_token = '</s>'
        self.max_seq_len = max_seq_len
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                         bos_token=self.bos_token, eos_token=self.eos_token, unk_token='<unk>',
                         pad_token='<pad>', mask_token='<mask>') 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data.iloc[index]
        document, label = str(record['document']), int(record['label'])
        tokens = [self.bos_token] + \
            self.tokenizer.tokenize(document) + [self.eos_token]
        encoder_input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(encoder_input_id) < self.max_seq_len:
            while len(encoder_input_id) < self.max_seq_len:
                encoder_input_id += [self.tokenizer.pad_token_id]
        else:
            encoder_input_id = encoder_input_id[:self.max_seq_len -
                                                1] + [self.tokenizer.eos_token_id]
        return {
            'text': np.array(encoder_input_id, dtype=np.int_),
            'label': np.array(label, dtype=np.int_)
        }


class NSMCDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_file,
                 test_file,
                 max_seq_len=128,
                 batch_size=32, num_workers=3):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=3,
                            help='number of workers for dataloader')
        return parser

    def setup(self, stage):
        # split dataset
        self.nsmc_train = NSMCDataset(self.train_file_path,
                                      self.max_seq_len)
        self.nsmc_test = NSMCDataset(self.test_file_path,
                                     self.max_seq_len)

    # return the dataloader for each split
    def train_dataloader(self):
        nsmc_train = DataLoader(self.nsmc_train,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True)
        return nsmc_train

    def val_dataloader(self):
        nsmc_val = DataLoader(self.nsmc_test,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              shuffle=False)
        return nsmc_val

    def test_dataloader(self):
        nsmc_test = DataLoader(self.nsmc_test,
                               batch_size=self.batch_size,
                               num_workers=self.num_workers,
                               shuffle=False)
        return nsmc_test
