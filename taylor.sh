#!/bin/bash

python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=/hdd3/jiayi/kaggle/train.txt \
    --processed-data-file=/hdd3/jiayi/kaggle/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=128 \
    --test-freq=1024 \
    --print-freq=1024 \
    --print-time \
    --test-mini-batch-size=16384 \
    --test-num-workers=16 \
    --arch-interaction-op cat \
    --load-model /hdd3/jiayi/result/baseline_cat_extended/best.pt \
    --mask_path ./masks/taylor/303166_0.5.pkl \
    --use-gpu \
    --gpu 6 \
    --nepochs 2 \
    2>&1 | tee taylor_303166_0.5.log

python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=/hdd3/jiayi/kaggle/train.txt \
    --processed-data-file=/hdd3/jiayi/kaggle/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=128 \
    --test-freq=1024 \
    --print-freq=1024 \
    --print-time \
    --test-mini-batch-size=16384 \
    --test-num-workers=16 \
    --arch-interaction-op cat \
    --load-model /hdd3/jiayi/result/baseline_cat_extended/best.pt \
    --mask_path ./masks/taylor/233023_0.4.pkl \
    --use-gpu \
    --gpu 7 \
    --nepochs 2 \
    2>&1 | tee taylor_233023_0.4.log

python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=/hdd3/jiayi/kaggle/train.txt \
    --processed-data-file=/hdd3/jiayi/kaggle/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=128 \
    --test-freq=1024 \
    --print-freq=1024 \
    --print-time \
    --test-mini-batch-size=16384 \
    --test-num-workers=16 \
    --arch-interaction-op cat \
    --load-model /hdd3/jiayi/result/baseline_cat_extended/best.pt \
    --mask_path ./masks/taylor/166000_0.3.pkl \
    --use-gpu \
    --gpu 2 \
    --nepochs 2 \
    2>&1 | tee taylor_166000_0.3.log

python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=/hdd3/jiayi/kaggle/train.txt \
    --processed-data-file=/hdd3/jiayi/kaggle/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=128 \
    --test-freq=1024 \
    --print-freq=1024 \
    --print-time \
    --test-mini-batch-size=16384 \
    --test-num-workers=16 \
    --arch-interaction-op cat \
    --load-model /hdd3/jiayi/result/baseline_cat_extended/best.pt \
    --mask_path ./masks/taylor/100089_0.2.pkl \
    --use-gpu \
    --gpu 0 \
    --nepochs 2 \
    2>&1 | tee taylor_100089_0.2.log

python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=/hdd3/jiayi/kaggle/train.txt \
    --processed-data-file=/hdd3/jiayi/kaggle/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=128 \
    --test-freq=1024 \
    --print-freq=1024 \
    --print-time \
    --test-mini-batch-size=16384 \
    --test-num-workers=16 \
    --arch-interaction-op cat \
    --load-model /hdd3/jiayi/result/baseline_cat_extended/best.pt \
    --mask_path ./masks/taylor/45012_0.1.pkl \
    --use-gpu \
    --gpu 0 \
    --nepochs 2 \
    2>&1 | tee taylor_45012_0.1.log

python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=/hdd3/jiayi/kaggle/train.txt \
    --processed-data-file=/hdd3/jiayi/kaggle/kaggleAdDisplayChallenge_processed.npz \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=128 \
    --test-freq=1024 \
    --print-freq=1024 \
    --print-time \
    --test-mini-batch-size=16384 \
    --test-num-workers=16 \
    --arch-interaction-op cat \
    --load-model /hdd3/jiayi/result/baseline_cat_extended/best.pt \
    --mask_path ./masks/taylor/34609_0.08.pkl \
    --use-gpu \
    --gpu 7 \
    --nepochs 2 \
    2>&1 | tee taylor_34609_0.08.log


