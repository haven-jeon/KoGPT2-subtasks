# KoGPT2-subtasks 

## KoGPT2 v2.0 한국어 평가 모듈

설치

```bash
git clone --recurse-submodules https://github.com/haven-jeon/KoGPT2-subtasks.git
cd KoGPT2-subtasks
pip install -r requirements 
```

## Subtasks

### NSMC

```bash
# sh run_nsmc.sh
CUDA_VISIBLE_DEVICES=0 python train.py \
                --batch_size 128 \
                --task nsmc \
                --train_data_path data/nsmc/ratings_train.txt \
                --val_data_path data/nsmc/ratings_test.txt \
                --gpus 1 \
                --seq_len 64 \
                --max_epochs 10
```

### KorSTS

```bash
# sh run_korsts.sh
CUDA_VISIBLE_DEVICES=0 python train.py \
                --batch_size 64 \
                --task korsts \
                --train_data_path data/KorNLUDatasets/KorSTS/sts-train.tsv \
                --val_data_path data/KorNLUDatasets/KorSTS/sts-test.tsv \
                --gpus 1 \
                --seq_len 64 \
                --max_epochs 5
```


### KorNLI

*Working*

