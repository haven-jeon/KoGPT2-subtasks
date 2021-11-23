import logging
import argparse

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from transformers.models.gpt2 import GPT2ForSequenceClassification
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import pytorch_lightning as pl


class Classification(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Classification, self).__init__()
        self.save_hyperparameters(hparams)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=32,
                            help='batch size for training (default: 96)')

        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.01
        }, {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr,
                          correct_bias=False)
        # warm up lr
        num_workers = (self.hparams.gpus if self.hparams.gpus > 0 else
                       1) * (self.hparams.num_nodes
                             if self.hparams.num_nodes is not None else 1)
        data_len = len(self.train_dataloader().dataset)
        logging.info(
            f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len /
                              (self.hparams.batch_size * num_workers *
                               self.hparams.accumulate_grad_batches) *
                              self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps)
        lr_scheduler = {
            'scheduler': scheduler,
            'monitor': 'loss',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]


class SubtaskGPT2(Classification):
    def __init__(self, hparams, **kwargs) -> None:
        super(SubtaskGPT2, self).__init__(hparams, **kwargs)
        self.model = GPT2ForSequenceClassification.from_pretrained(
            'skt/kogpt2-base-v2',
            num_labels=self.hparams.num_labels)
        self.metric_acc = pl.metrics.classification.Accuracy()
        self.metric_acc_val = pl.metrics.classification.Accuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, batch):
        input_ids = batch["text"]
        attention_mask = input_ids.ne(self.model.config.pad_token_id).float()

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return logits

    def training_step(self, batch, batch_idx):
        label = batch["label"]
        output = self(batch)
        y_hat = output['logits']

        train_loss = self.loss_function(y_hat, label)
        train_acc = self.metric_acc(torch.nn.functional.softmax(y_hat, dim=1),
                                    label)
        self.log("train_loss", train_loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", train_acc.compute(), on_step=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        label = batch["label"]
        output = self(batch)
        y_hat = output['logits']

        val_loss = self.loss_function(y_hat, label)
        val_acc = self.metric_acc_val(torch.nn.functional.softmax(y_hat, dim=1),
                                      label)
        self.log("loss", val_loss, on_epoch=True, prog_bar=True)
        self.log('val_acc_step', val_acc, on_step=True, on_epoch=True)

        return {"loss": val_loss, "batch_cnt": label.shape[0]}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        self.log('val_acc',
                 self.metric_acc_val.compute(),
                 on_epoch=True,
                 prog_bar=True)


class SubtaskGPT2Regression(Classification):
    def __init__(self, hparams, **kwargs) -> None:
        super(SubtaskGPT2Regression, self).__init__(hparams, **kwargs)
        self.model = GPT2ForSequenceClassification.from_pretrained(
            'skt/kogpt2-base-v2',
            num_labels=self.hparams.num_labels)

    def forward(self, batch):
        input_ids = batch["text"]
        attention_mask = input_ids.ne(self.model.config.pad_token_id).float()
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)['logits']
        input_ids_rev = batch["text_rev"]
        logits += self.model(input_ids=input_ids_rev, attention_mask=attention_mask)['logits']
        return logits

    def training_step(self, batch, batch_idx):
        label = batch["label"]
        y_hat = self(batch)
        loss = torch.nn.functional.mse_loss(y_hat.squeeze(), label.float())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        label = batch["label"]
        result = {'y_hat': y_hat.cpu().numpy(),
                  'label': label.cpu().numpy()}
        return result

    def validation_epoch_end(self, outputs):
        y_hats = np.concatenate([x["y_hat"] for x in outputs]).squeeze()
        labels = np.concatenate([x["label"] for x in outputs])
        pearson_corr = pearsonr(y_hats, labels)[0]
        spearman_corr = spearmanr(y_hats, labels)[0]        
        self.log('pearson_corr', pearson_corr, prog_bar=True)
        self.log('spearman_corr', spearman_corr, prog_bar=True)
