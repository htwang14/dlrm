import numpy as np 
import os, argparse, pickle

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
mask = (score > score_th)
print('mask:', mask.shape, mask[0:10])

masks = np.array_split(mask, split_ids)
mask_dict = {}
for mask, (name, _) in zip(masks, importance_score.items()):
    print(name, mask.shape)
    mask_dict[name] = mask
    print('mask:', mask.shape)

# find flops
c1 = np.sum(masks[2])
c2 = np.sum(masks[1])
FLOPs = int(2*432*c1 + c1 + 2*c1*c2 + c2 + 2*c2*1 + 1)
print(c1, c2, FLOPs/705793)
 
# save mask
save_dir = os.path.join('masks', args.metric)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
f = open(os.path.join(save_dir, '%d_%s.pkl' % (FLOPs, args.ratio)), "wb")
pickle.dump(mask_dict, f)
f.close()
