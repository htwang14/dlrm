#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python dlrm_s_pytorch.py \
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
    --load-model results/taylor/309550_0.5/model.pth \
    --mask_path ./masks/taylor/309550_0.5.pkl \
    --use-gpu \
    --nepochs 2 \
    2>&1 | tee taylor_309550_0.5.log

CUDA_VISIBLE_DEVICES=0 python dlrm_s_pytorch.py \
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
    --mask_path ./masks/taylor/241999_0.4.pkl \
    --use-gpu \
    --nepochs 2 \
    2>&1 | tee taylor_241999_0.4.log

CUDA_VISIBLE_DEVICES=0 python dlrm_s_pytorch.py \
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
    --mask_path ./masks/taylor/177590_0.3.pkl \
    --use-gpu \
    --nepochs 2 \
    2>&1 | tee taylor_177590_0.3.log

CUDA_VISIBLE_DEVICES=0 python dlrm_s_pytorch.py \
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
    --mask_path ./masks/taylor/116937_0.2.pkl \
    --use-gpu \
    --nepochs 2 \
    2>&1 | tee taylor_116937_0.2.log

CUDA_VISIBLE_DEVICES=0 python dlrm_s_pytorch.py \
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
    --mask_path ./masks/taylor/58306_0.1.pkl \
    --use-gpu \
    --nepochs 2 \
    2>&1 | tee taylor_58306_0.1.log

CUDA_VISIBLE_DEVICES=0 python dlrm_s_pytorch.py \
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
    --mask_path ./masks/taylor/28822_0.05.pkl \
    --use-gpu \
    --nepochs 2 \
    2>&1 | tee taylor_28822_0.05.log


