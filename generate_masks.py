import numpy as np 
import os, argparse, pickle, torch

parser = argparse.ArgumentParser(description="Train Deep Learning Recommendation Model (DLRM)")
parser.add_argument("--ratio", '-r', type=float)
parser.add_argument("--metric", '-m', choices=['l1', 'taylor'])
args = parser.parse_args()

importance_score = pickle.load(open(os.path.join('importance_score', '%s.pkl' % args.metric), "rb"))
print(type(importance_score))
# print(importance_score)
masks = {}
split_ids = []
C = 0
score_list = []
for name, score in importance_score.items():
    C += score.shape[0]
    split_ids.append(C)
    score_list.append(score)
score = np.concatenate(score_list)
print('split_ids:', split_ids)
print('score:', score.shape, score[0:20])
score_sorted = np.sort(score)[::-1]
score_th = score_sorted[int(args.ratio * len(score))]
# print('score_sorted:', score_sorted)
print('score_th:', score_th)
prune_bin = (score <= score_th) # 1: prune; 0: keep
print('prune_bin:', prune_bin.shape, prune_bin[0:10])

prune_bins = np.array_split(prune_bin, split_ids)

prune_bins_w3 = prune_bins[0]
prune_bins_w2 = prune_bins[1]
prune_bins_w1 = prune_bins[2]
print('prune_bins_w3:', prune_bins_w3.shape)
print('prune_bins_w2:', prune_bins_w2.shape)
print('prune_bins_w1:', prune_bins_w1.shape)
m1 = np.ones((512, 432))
m1[prune_bins_w1,:] = 0
m2 = np.ones((256, 512))
m2[prune_bins_w2,:] = 0
m2[:,prune_bins_w1] = 0
m3 = np.ones((1, 256))
m3[:,prune_bins_w2] = 0
mask_list = [torch.from_numpy(m1), torch.from_numpy(m2), torch.from_numpy(m3)]

# find flops
c1 = 512-np.sum(prune_bins_w1)
c2 = 256-np.sum(prune_bins_w2)
FLOPs = int(2*432*c1 + c1 + 2*c1*c2 + c2 + 2*c2*1 + 1)
print(c1, c2, FLOPs, FLOPs/705793)
 
# save mask
save_dir = os.path.join('masks', args.metric)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
f = open(os.path.join(save_dir, '%d_%s.pkl' % (FLOPs, args.ratio)), "wb")
pickle.dump(mask_list, f)
f.close()

f = open(os.path.join(save_dir, '%d_%s.pkl' % (FLOPs, args.ratio)), "rb")
mask_list = pickle.load(f)
for mask in mask_list:
    print(mask.shape)
f.close()
