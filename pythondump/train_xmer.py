#!/usr/bin/env python3
# train.py – FT-Transformer + stable NT-Xent for PDW embeddings
import os, json, random, functools, builtins, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

# ──────────────────────────── Config ────────────────────────────
EMBED_DIM      = 128          # output size of encoder
BATCH_GPU      = 256          # per-GPU batch; world-batch = BATCH_GPU × #GPUs
EPOCHS         = 60
BASE_LR        = 3e-3         # will be multiplied by world size
WARMUP_EPOCHS  = 2
TEMP           = 0.1          # NT-Xent temperature
CENTER_WT      = 0.1
CLUSTER_EVERY  = 5
RESULT_DIR     = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ───────────────────── Distributed helpers ──────────────────────
def setup_dist():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world

def is_main(r): return r == 0
def mute_other_ranks(rank):
    if not is_main(rank):
        builtins.print = functools.partial(lambda *a, **k: None)

# ───────────────────────── Data utils ───────────────────────────
COLUMNS = ['Name', 'PW(µs)', 'Azimuth(º)', 'Elevation(º)',
           'Power(dBm)', 'Freq(MHz)']
FEATS, LABEL = COLUMNS[1:], 'Name'

def preprocess(df, scaler=None):
    x = df[FEATS].values.astype(np.float32)
    y, uniques = pd.factorize(df[LABEL])
    if scaler is None:
        scaler = RobustScaler(); x = scaler.fit_transform(x)
    else:
        x = scaler.transform(x)
    return x, y, {i: n for i, n in enumerate(uniques)}, scaler

class PDW(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

class ClassBalancedSampler(DistributedSampler):
    """
    Guarantees ≥1 positive per sample in every mini-batch.
    """
    def __init__(self, labels, num_replicas=None, rank=None,
                 per_class=2, shuffle=True):
        self.labels = labels
        self.per_class = per_class
        super().__init__(labels, num_replicas, rank, shuffle)

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.epoch)
        lbl2idx = defaultdict(list)
        for idx, l in enumerate(self.labels):
            lbl2idx[int(l)].append(idx)

        # shuffle indices inside each class
        for l in lbl2idx:
            idxs = torch.tensor(lbl2idx[l])
            lbl2idx[l] = idxs[torch.randperm(len(idxs), generator=rng)].tolist()

        # pack groups of per_class
        flat = []
        for idxs in lbl2idx.values():
            for i in range(0, len(idxs), self.per_class):
                flat.extend(idxs[i:i+self.per_class])

        # pad to multiple of world size
        total = int(np.ceil(len(flat)/self.num_replicas))*self.num_replicas
        if len(flat) < total:
            flat += flat[: total-len(flat)]

        # shard
        return iter(flat[self.rank::self.num_replicas])

# ────────────────── FT-Transformer encoder ──────────────────────
class FTTransformer(nn.Module):
    def __init__(self, num_feats, dim, heads=8, layers=3, drop=0.2):
        super().__init__()
        self.token = nn.Parameter(torch.randn(num_feats, dim))
        self.cls   = nn.Parameter(torch.randn(1, dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dropout=drop, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):                 # x (B, F)
        B = x.size(0)
        tokens = self.token * x.unsqueeze(-1)          # (B, F, D)
        tokens = torch.cat([self.cls.expand(B, -1, -1), tokens], dim=1)
        out = self.encoder(tokens)[:, 0]               # CLS token
        return nn.functional.normalize(self.norm(out), p=2, dim=1)

# ─────────────── Stable NT-Xent & CenterLoss ────────────────────
def nt_xent(z, y, τ=0.1, eps=1e-8):
    """
    Numerically-stable NT-Xent without any inplace ops.
    z : (B,D) L2-normalised embeddings
    y : (B,)  labels
    """
    # pairwise cosine similarity
    sim = z @ z.t() / τ                           # (B,B)

    # mask self-similarity ***without*** inplace writes
    self_mask = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(self_mask, -float("inf"))   # NEW – returns a copy

    # log-softmax (row-wise) – stable
    max_row = sim.max(dim=1, keepdim=True).values
    log_soft = (sim - max_row).exp()
    log_soft = log_soft / log_soft.sum(dim=1, keepdim=True)
    log_soft = (log_soft + eps).log()

    # positives mask
    pos = (y[:, None] == y[None, :]).float()
    pos = pos.masked_fill(self_mask, 0)           # avoids diag write

    denom = pos.sum(1) + eps
    loss = -(log_soft * pos).sum(1) / denom
    return loss.mean()


class CenterLoss(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, dim))
    def forward(self, f, y):
        return (f - self.centers[y]).pow(2).sum(1).mean()

# ───────────────── Clustering evaluation ───────────────────────
def cluster_acc(y, pred):
    cm = pd.crosstab(pred, y)
    r, c = linear_sum_assignment(-cm.values)
    return cm.values[r, c].sum() / len(y)

@torch.no_grad()
def evaluate(model, x_test, y_test):
    model.eval()
    emb = model(torch.tensor(x_test).cuda()).cpu().numpy()
    k = len(np.unique(y_test))
    pred = KMeans(k, n_init=10, random_state=42).fit_predict(emb)
    return cluster_acc(y_test, pred)

# ─────────────────────────── main ──────────────────────────────
def main():
    rank, world = setup_dist()
    mute_other_ranks(rank)

    # ─── Data loading
    df1 = pd.read_excel('set1.xls')[COLUMNS]
    df2 = pd.read_excel('set2.xls')[COLUMNS]
    df3 = pd.read_excel('set3.xlsx')[COLUMNS]
    df5 = pd.read_excel('set5.xlsx')[COLUMNS]
    df6 = pd.read_excel('set6.xlsx')[COLUMNS]

    train_df = pd.concat([df1, df2, df5, df6], ignore_index=True)
    test_df  = df3

    x_tr, y_tr, lbl_map, scaler = preprocess(train_df)
    x_te, y_te, _, _            = preprocess(test_df, scaler)

    ds = PDW(x_tr, y_tr)
    sampler = ClassBalancedSampler(y_tr)
    dl = DataLoader(ds, batch_size=BATCH_GPU,
                    sampler=sampler, num_workers=4,
                    pin_memory=True, drop_last=True)

    # ─── Model & losses
    model = FTTransformer(len(FEATS), EMBED_DIM).cuda()
    model = DDP(model, device_ids=[rank], output_device=rank)

    center_loss = CenterLoss(len(lbl_map), EMBED_DIM).cuda()
    lr = BASE_LR * world
    opt = optim.AdamW([*model.parameters(), *center_loss.parameters()], lr=lr)
    warm = LambdaLR(opt, lambda e: min(1, (e+1)/WARMUP_EPOCHS))
    cosine = CosineAnnealingLR(opt, EPOCHS - WARMUP_EPOCHS)

    if is_main(rank):
        print(f"[Rank0] GPUs={world}  batch/GPU={BATCH_GPU}  LR={lr}")

    # ─── Training loop
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        running = 0.0

        for x, y in dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            opt.zero_grad()
            f = model(x)
            loss = nt_xent(f, y) + CENTER_WT * center_loss(f, y)
            if torch.isnan(loss):
                raise RuntimeError("Loss became NaN – check sampler.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()

        # schedulers
        if epoch < WARMUP_EPOCHS: warm.step()
        else:                     cosine.step()

        if (epoch + 1) % CLUSTER_EVERY == 0 and is_main(rank):
            acc = evaluate(model.module, x_te, y_te)
            print(f"Epoch {epoch+1:3d}  loss {running/len(dl):.4f}  "
                  f"clust-acc {acc*100:5.2f}%")

    # ─── Final test & save
    if is_main(rank):
        acc = evaluate(model.module, x_te, y_te)
        json.dump({"embed_dim": EMBED_DIM,
                   "epochs": EPOCHS,
                   "clustering_acc": acc},
                  open(f"{RESULT_DIR}/result_{EMBED_DIM}.json", "w"), 2)
        print(f"[Done] Final clustering accuracy {acc*100:.2f}%")

    dist.destroy_process_group()

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
