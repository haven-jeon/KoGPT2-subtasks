import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import PreTrainedTokenizerFast


class KorSTSDataset(Dataset):
    def __init__(self, filepath, max_seq_len=128):
        self.filepath = filepath
        self.data = pd.read_csv(self.filepath, sep='\t', quoting=3)
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
        q1, q2, label = str(record['sentence1']), str(
            record['sentence2']), float(record['score'])
        tokens = [self.bos_token]  + self.tokenizer.tokenize(q1) + [self.eos_token, self.bos_token] + \
                 self.tokenizer.tokenize(q2) + [self.eos_token + '<unused0>']
        rev_tokens = [self.bos_token] + self.tokenizer.tokenize(q2) + [self.eos_token,self.bos_token] + \
                     self.tokenizer.tokenize(q1) + [self.eos_token + '<unused0>']
        encoder_input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        encoder_input_id_rev = self.tokenizer.convert_tokens_to_ids(rev_tokens)
        if len(encoder_input_id) < self.max_seq_len:
            while len(encoder_input_id) < self.max_seq_len:
                encoder_input_id += [self.tokenizer.pad_token_id]
                encoder_input_id_rev += [self.tokenizer.pad_token_id]
        else:
            encoder_input_id = encoder_input_id[:self.max_seq_len -
                                                1] + [self.tokenizer.eos_token_id]
            encoder_input_id_rev = encoder_input_id_rev[:self.max_seq_len -
                                                        1] + [
                                                            self.tokenizer.eos_token_id
                                                        ]
        return {
            'text': np.array(encoder_input_id, dtype=np.int_),
            'text_rev': np.array(encoder_input_id_rev, dtype=np.int_),
            'label': np.array(label, dtype=np.float)
        }


class KorSTSDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_file,
                 test_file,
                 max_seq_len=128,
                 batch_size=32,
                 num_workers=3):
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
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.ksts_train = KorSTSDataset(self.train_file_path,
                                        self.max_seq_len)
        self.ksts_test = KorSTSDataset(self.test_file_path,
                                       self.max_seq_len)

    # return the dataloader for each split
    def train_dataloader(self):
        ksts_train = DataLoader(self.ksts_train,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True)
        return ksts_train

    def val_dataloader(self):
        ksts_val = DataLoader(self.ksts_test,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              shuffle=False)
        return ksts_val

    def test_dataloader(self):
        ksts_test = DataLoader(self.ksts_test,
                               batch_size=self.batch_size,
                               num_workers=self.num_workers,
                               shuffle=False)
        return ksts_test