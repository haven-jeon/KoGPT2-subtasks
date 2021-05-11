CUDA_VISIBLE_DEVICES=0 python train.py --max_epochs 5 \
                --batch_size 64 \
                --task korsts \
                --train_data_path data/KorNLUDatasets/KorSTS/sts-train.tsv \
                --val_data_path data/KorNLUDatasets/KorSTS/sts-test.tsv \
                --gpus 1 \
                --seq_len 64 \
                --max_epochs 10


