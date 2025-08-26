#!/usr/bin/env python3
# train_ft_triplet_deep.py – deep FT-Transformer + semi-hard TripletLoss
import os, json, random, functools, builtins, warnings
from collections import defaultdict
import numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

import torch, torch.nn as nn, torch.optim as optim, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

# ───────────────────────── Config ─────────────────────────
EMBED_DIM      = 192
LAYERS         = 6
HEADS          = 8
BATCH_GPU      = 128           # anchors per GPU
MARGIN         = 0.3
EPOCHS         = 80
BASE_LR        = 3e-3          # scaled by world
CLUST_EVERY    = 10
RESULT_DIR     = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ─────────────────── Distributed helpers ──────────────────
def setup_dist():
    dist.init_process_group("nccl")
    r, w = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(r); return r, w

def is_main(r): return r == 0
def mute(rank): 
    if not is_main(rank):
        builtins.print = functools.partial(lambda *a, **k: None)

# ───────────────────── Data utilities ─────────────────────
COLS = ['Name','PW(µs)','Azimuth(º)','Elevation(º)','Power(dBm)','Freq(MHz)']
FEATS, LABEL = COLS[1:], 'Name'

def preprocess(df, scaler=None):
    x = df[FEATS].values.astype(np.float32)
    y, _ = pd.factorize(df[LABEL])
    if scaler is None:
        scaler = RobustScaler(); x = scaler.fit_transform(x)
    else:
        x = scaler.transform(x)
    return x, y, scaler

class TripletPDW(Dataset):
    """Return (a,p,n) **and** their original indices for label gathering."""
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.lbl2idx = defaultdict(list)
        for i,l in enumerate(y): self.lbl2idx[int(l)].append(i)
        if len(self.lbl2idx) < 2:
            raise ValueError("Need ≥2 classes.")
    def __len__(self): return len(self.x)
    def __getitem__(self, idx):
        a_idx = idx
        p_idx = random.choice(self.lbl2idx[self.y[a_idx]])
        neg_lbl = random.choice([l for l in self.lbl2idx if l != self.y[a_idx]])
        n_idx = random.choice(self.lbl2idx[neg_lbl])
        return (torch.from_numpy(self.x[a_idx]),
                torch.from_numpy(self.x[p_idx]),
                torch.from_numpy(self.x[n_idx]),
                a_idx, p_idx, n_idx)

# ──────────────── FT-Transformer encoder ────────────────
class FTTransformer(nn.Module):
    def __init__(self, feats, dim=EMBED_DIM, heads=HEADS,
                 layers=LAYERS, drop=0.2):
        super().__init__()
        self.token = nn.Parameter(torch.randn(feats, dim))
        self.cls   = nn.Parameter(torch.randn(1, dim))
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                         dropout=drop, batch_first=True)
        self.tf   = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):                       # x (B,F)
        B = x.size(0)
        tok = self.token * x.unsqueeze(-1)
        tok = torch.cat([self.cls.expand(B,-1,-1), tok], 1)
        out = self.tf(tok)[:,0]
        return nn.functional.normalize(self.norm(out), p=2, dim=1)

# ───────────── Semi-hard in-batch triplet ───────────────
def semi_hard_triplet(feat, lbl, margin=MARGIN):
    # cosine distance because feats are L2-normed
    dist = 1 - feat @ feat.t()
    pos_mask = (lbl[:,None] == lbl[None,:]).bool()
    neg_mask = ~pos_mask

    pos_d = dist.masked_fill(~pos_mask, 1e9).min(1).values        # hardest +
    bigger = dist + (pos_d[:,None] - dist) * (~neg_mask)          # filter
    neg_d = bigger.masked_fill(~neg_mask, 1e9).min(1).values      # semi-hard −

    loss = nn.functional.relu(pos_d - neg_d + margin)
    return loss.mean()

# ───────────────── Clustering metric ────────────────────
def clust_acc(y, pred):
    tab = pd.crosstab(pred, y)
    r,c = linear_sum_assignment(-tab.values)
    return tab.values[r,c].sum()/len(y)

@torch.no_grad()
def evaluate(net, x, y):
    net.eval()
    emb = net(torch.tensor(x).cuda()).cpu().numpy()
    k   = len(np.unique(y))
    cid = KMeans(k, n_init=10, random_state=42).fit_predict(emb)
    return clust_acc(y, cid)

# ─────────────────────────── main ───────────────────────
def main():
    rank, world = setup_dist(); mute(rank)

    # Load datasets
        # Load datasets
    df1 = pd.read_excel('set1.xls')
    df1 = df1[df1['Status'] != 'DELETE_EMITTER'][COLS]
    
    df2 = pd.read_excel('set2.xls')
    df2 = df2[df2['Status'] != 'DELETE_EMITTER'][COLS]
    
    df3 = pd.read_excel('set3.xlsx')
    df3 = df3[df3['Status'] != 'DELETE_EMITTER'][COLS]
    
    df5 = pd.read_excel('set5.xlsx')
    df5 = df5[df5['Status'] != 'DELETE_EMITTER'][COLS]
    
    df6 = pd.read_excel('set6.xlsx')
    df6 = df6[df6['Status'] != 'DELETE_EMITTER'][COLS]

    x_tr,y_tr,sc = preprocess(pd.concat([df1,df2,df5,df6],ignore_index=True))
    x_te,y_te,_  = preprocess(df3, sc)

    ds = TripletPDW(x_tr, y_tr)
    sampler = DistributedSampler(ds, shuffle=True)
    dl = DataLoader(ds, batch_size=BATCH_GPU, sampler=sampler,
                    num_workers=4, pin_memory=True, drop_last=True)

    # Model & optimiser
    model = FTTransformer(len(FEATS)).cuda()
    model = DDP(model, device_ids=[rank], output_device=rank)
    opt   = optim.AdamW(model.parameters(), lr=BASE_LR*world)
    sched = CosineAnnealingLR(opt, EPOCHS)

    if is_main(rank):
        print(f"[Rank0] GPUs={world}  batch/GPU={BATCH_GPU}  LR={BASE_LR*world}")

    # Training loop
    for ep in range(EPOCHS):
        sampler.set_epoch(ep); model.train(); run = 0.0
        for a,p,n,idx_a,idx_p,idx_n in dl:
            a,p,n = a.cuda(), p.cuda(), n.cuda()
            opt.zero_grad()

            feats = torch.cat([model(a), model(p), model(n)], 0)
            batch_lbl = torch.tensor(
                np.concatenate([y_tr[idx_a], y_tr[idx_p], y_tr[idx_n]]),
                device='cuda')
            loss = semi_hard_triplet(feats, batch_lbl)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); run += loss.item()

        sched.step()

        if (ep+1) % CLUST_EVERY == 0 and is_main(rank):
            acc = evaluate(model.module, x_te, y_te)
            print(f"Epoch {ep+1:3d}  loss {run/len(dl):.4f}  cl-acc {acc*100:5.2f}%")

    # Final eval & save
    if is_main(rank):
        acc = evaluate(model.module, x_te, y_te)
        json.dump({"embed_dim":EMBED_DIM,"layers":LAYERS,"margin":MARGIN,
                   "epochs":EPOCHS,"clustering_acc":acc},
                  open(f"{RESULT_DIR}/ft_triplet_{EMBED_DIM}.json","w"),2)
        print(f"[Done] final clustering {acc*100:.2f}%")

    dist.destroy_process_group()

# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
