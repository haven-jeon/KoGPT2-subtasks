import argparse


class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_data_path',
                            type=str,
                            default=None,
                            help='train data path')

        parser.add_argument('--output_dir',
                            type=str,
                            default=None,
                            help='directory to collect tensorboard logs and params')

        parser.add_argument('--val_data_path',
                            type=str,
                            default=None,
                            help='valid data path')

        parser.add_argument('--seq_len',
                            type=int,
                            default=512,
                            help='valid data path')

        parser.add_argument('--batch_size',
                            type=int,
                            default=32,
                            help='number of batch size')

        parser.add_argument('--seed',
                            type=int,
                            default=7874,
                            help='random seed')
        return parser
