## baseline: hand-crafted 
See `baseline.sh`

## baseline: l1 pruning & Taylor pruning
First run `oneshot.py` (see `oneshot.sh`) to generate importance score of each channel.
Then run `generate_masks.py` to generate binary masks (saved in `./masks`).
Then run `baseline.py` (See `baseline.sh`) to finetune on subnetworks.

