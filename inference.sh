CUDA_VISIBLE_DEVICES=7 python inference.py \
    --arch-sparse-feature-size=16 \
    --arch-mlp-bot="13-512-256-64-16" \
    --arch-mlp-top="512-256-1" \
    --data-generation=dataset \
    --data-set=kaggle \
    --raw-data-file=/hdd3/jiayi/kaggle/train.txt \
    --processed-data-file=/hdd3/jiayi/kaggle/kaggleAdDisplayChallenge_processed.npz \
    --load-model /hdd3/jiayi/result/baseline_cat_extended/best.pt \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=128 \
    --print-freq=1024 \
    --print-time \
    --test-mini-batch-size=16384 \
    --test-num-workers=16 \
    --test-freq=1024 \
    --arch-interaction-op cat \
    --use-gpu \
    --inference-only