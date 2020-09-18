#!/bin/bash

# 432 -> 512 -> 256- > 1
# FLOPs = 2*432*512+512+2*512*256+256+2*256+1 = 705793

# FLOPs = 2*432*300+300+2*300*150+150+2*150+1 = 349951
# CR=0.4958 best 78.797 %
python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="300-150-1" \
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
    2>&1 | tee 300-150-1_lr0.1.log

# CR=40.7%
python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="256-128-1" \
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
    2>&1 | tee 256-128-1_lr0.1.log

# FLOPs = 2*432*240+240+2*240*120+2*120+120+1 = 265561
# CR=0.3763 best 78.797 %
python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="240-120-1" \
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
    2>&1 | tee 240-120-1_lr0.1.log

# FLOPs = 2*432*224+224+2*224*112+112+2*112+1 = 122305
# CR=0.3439 best 78.777 %
python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="224-112-1" \
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
    2>&1 | tee 224-112-1_lr0.1.log

# FLOPs = 2*432*192+192+2*192*96+96+2*96+1 = 203233
# CR=28.79%
CUDA_VISIBLE_DEVICES=2 python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="192-96-1" \
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
    --load-model results/192-96-1_lr0.1/model.pth \
    --use-gpu \
    2>&1 | tee 192-96-1_lr0.1.log

# FLOPs = 2*432*128+128+2*128*64+64+2*64+1 = 127297
# CR=18.04%
CUDA_VISIBLE_DEVICES=2 python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="128-64-1" \
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
    --use-gpu \
    2>&1 | tee 128-64-1_lr0.1.log

# FLOPs = 2*432*64+64+2*64*32+32+2*32+1 = 59553
# CR=8.44%
CUDA_VISIBLE_DEVICES=2 python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="64-32-1" \
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
    --load-model results/64-32-1_lr0.1/model.pth \
    --use-gpu \
    2>&1 | tee 64-32-1_lr0.1.log

# FLOPs = 2*432*48+48+2*48*24+24+2*24+1 = 43897
# CR=8.44%
CUDA_VISIBLE_DEVICES=2 python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="48-24-1" \
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
    --use-gpu \
    2>&1 | tee 48-24-1_lr0.1.log

# FLOPs = 2*432*40+40+2*40*20+20+2*20+1 = 43897
# CR=8.44%
CUDA_VISIBLE_DEVICES=1 python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="40-20-1" \
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
    --use-gpu \
    2>&1 | tee 40-20-1_lr0.1.log

# FLOPs = 2*432*32+32+2*32*16+16+2*16+1 = 28753
# CR=4.07%
CUDA_VISIBLE_DEVICES=2 python dlrm_s_pytorch.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="32-16-1" \
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
    --load-model results/32-16-1_lr0.1/model.pth \
    --use-gpu \
    2>&1 | tee 32-16-1_lr0.1.log
