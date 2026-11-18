"""Train matched binary detectors and select checkpoints by validation BCE."""
import copy
from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
import torch
import torch.nn.functional as F

from src.model import IoMTMLP


def make_model(cfg):
    w1,w2 = cfg["widths"]
    model = IoMTMLP(45)
    model.net = torch.nn.Sequential(torch.nn.Linear(45,w1),torch.nn.ReLU(),
        torch.nn.Dropout(cfg["dropout"]),torch.nn.Linear(w1,w2),torch.nn.ReLU(),
        torch.nn.Linear(w2,1),torch.nn.Sigmoid())
    return model


def as_tensor(part):
    return torch.from_numpy(part["x"]),torch.tensor(part["y"],dtype=torch.float32)


def logits(model, x):
    model.eval()
    with torch.inference_mode():
        return model.net[:-1](torch.as_tensor(x)).flatten().numpy()


def proximal_penalty(parameters, anchor, mu):
    return mu/2 * sum((p-q).square().sum() for p,q in zip(parameters,anchor))


def average(models, counts):
    with torch.no_grad():
        return [sum(w*p for w,p in zip(counts,parts))/sum(counts)
                for parts in zip(*(model.parameters() for model in models))]


def fit(model, data, optimizer, cfg, epochs, rng_seed, mu=0, balance=False, steps=None):
    x,y = data
    torch.manual_seed(rng_seed)
    model.train()
    anchor = [p.detach().clone() for p in model.parameters()] if mu else None
    total,start,n_steps = 0.,perf_counter(),0
    if balance:
        fractions = y.mean()
        weights = torch.where(y==1,0.5/fractions,0.5/(1-fractions))
    for _ in range(epochs):
        for ids in torch.randperm(len(y)).split(cfg["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            predicted = model.net[:-1](x[ids]).flatten()
            loss = F.binary_cross_entropy_with_logits(predicted,y[ids],weight=weights[ids] if balance else None)
            total += loss.detach().item()
            if anchor is not None:
                loss += proximal_penalty(model.parameters(),anchor,mu)
            loss.backward()
            optimizer.step()
            n_steps += 1
            if n_steps == steps:
                return {"training_bce":total/n_steps,"steps":n_steps,"seconds":perf_counter()-start}
    return {"training_bce":total/n_steps,"steps":n_steps,"seconds":perf_counter()-start}


def optimizer(model, cfg):
    return torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=cfg["lr"],fused=True)


def vectors(models, starts, members):
    return {k:torch.cat([(p-q).flatten() for p,q in zip(models[k].parameters(),starts[k])]).detach().numpy()
            for k in members}


def update_cosine(a,b):
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(np.dot(a,b)/norm,-1,1)) if norm else None


def split_cluster(members, deltas, initial_norms, cfg, round_id):
    """Sattler norm gate and complete linkage on pairwise cosine distances."""
    if len(members)<2 or round_id<cfg["min_round"]:
        return [members]
    update = np.stack([deltas[k] for k in members])
    scale = max(initial_norms[k] for k in members)
    mean_norm = np.linalg.norm(update.mean(axis=0))
    max_norm = np.linalg.norm(update,axis=1).max()
    if scale == 0 or mean_norm >= scale*cfg["mean_norm_fraction"] or max_norm <= scale*cfg["max_norm_fraction"]:
        return [members]
    dist = np.zeros((len(members),len(members)))
    for i,j in combinations(range(len(members)),2):
        cosine = update_cosine(update[i],update[j])
        if cosine is None:
            return [members]
        dist[i,j]=dist[j,i]=1-cosine
    partition = fcluster(linkage(squareform(dist),method="complete"),2,criterion="maxclust")
    return [[k for k,g in zip(members,partition) if g==i] for i in np.unique(partition)]


def train(clients, cfg, seed, method, mu=0, head_depth=1, capture_round=None):
    torch.manual_seed(seed)
    names = list(clients)
    base = make_model(cfg)
    models = {k:copy.deepcopy(base) for k in names}
    xys = {k:as_tensor(clients[k]["train"]) for k in names}
    val = {k:as_tensor(clients[k]["validation"]) for k in names}
    counts = {k:len(y) for k,(_,y) in xys.items()}
    grouped = [[k] for k in names] if method=="local_only" else [names]
    if method=="fixed_cluster":
        grouped = cfg["fixed_groups"]
    head_start = 4 if head_depth==1 else 2
    rounds,epochs = cfg["rounds"],cfg["local_epochs"]
    persistent = {k:optimizer(models[k],cfg) for k in names}
    pooled = method in ("centralized","centralized_matched")
    if pooled:
        pooled_opt = optimizer(base,cfg)
        all_train = tuple(torch.cat(v) for v in zip(*xys.values()))
        rounds,epochs = cfg["pooled_epochs"],1
        if method=="centralized_matched":
            rounds = cfg["rounds"]
    initial_norms, history, best, best_loss, best_round = {}, [], None, float("inf"), None
    started = perf_counter()
    snapshot = None
    for r in range(1,rounds+1):
        update_pairs = {}
        if pooled:
            nsteps = None
            if method=="centralized_matched":
                nsteps = cfg["local_epochs"]*sum((n+cfg["batch_size"]-1)//cfg["batch_size"] for n in counts.values())
            stat = fit(base,all_train,pooled_opt,cfg,epochs if nsteps is None else cfg["local_epochs"]+1,
                       seed+r*1000,steps=nsteps)
            if nsteps is not None and stat["steps"] != nsteps:
                raise ValueError("Matched central budget was not reached")
            round_train = {"pooled":stat}
            models = {k:base for k in names}
        else:
            starts = {k:[p.detach().clone() for p in models[k].parameters()] for k in names}
            round_train = {}
            for i,k in enumerate(names):
                opt = persistent[k] if method=="local_only" else optimizer(models[k],cfg)
                round_train[k] = fit(models[k],xys[k],opt,cfg,epochs,seed+r*1000+i,
                                     mu if method=="fedprox" else 0,method=="class_balanced")
            deltas = vectors(models,starts,names)
            update_pairs = {a+"-"+b:update_cosine(deltas[a],deltas[b])
                            for members in grouped for a,b in combinations(members,2)}
            if method=="adaptive_cluster":
                if r==1:
                    initial_norms = {k:float(np.linalg.norm(deltas[k])) for k in names}
                grouped = [g for members in grouped
                           for g in split_cluster(members,deltas,initial_norms,cfg["adaptive"],r)]
            if method!="local_only":
                for members in grouped:
                    shared = average([models[k] for k in members],
                                     [1 if method=="equal_weight" else counts[k] for k in members])
                    if method=="fedper":
                        shared = shared[:head_start]
                    with torch.no_grad():
                        for k in members:
                            for p,q in zip(models[k].parameters(),shared):
                                p.copy_(q)
        validation, va_accuracy = {}, {}
        for k,(x,y) in val.items():
            score_k = logits(models[k],x)
            validation[k] = float(F.binary_cross_entropy_with_logits(torch.from_numpy(score_k),y))
            va_accuracy[k] = float(np.mean((score_k>=0)==y.numpy()))
        score = np.mean(list(validation.values()))
        if score < best_loss:
            best_loss,best_round = score,r
            best = {k:copy.deepcopy(models[k].state_dict()) for k in names}
        history.append({"round":r,"train":round_train,"validation_bce":validation,
                        "validation_accuracy":va_accuracy,"groups":copy.deepcopy(grouped),"update_cosine":update_pairs})
        if r==capture_round:
            snapshot={k:copy.deepcopy(models[k].state_dict()) for k in names}
    for k in names:
        models[k].load_state_dict(best[k])
    return models, {"history":history,"selected_round":best_round,"seconds":perf_counter()-started,
                    "optimizer_steps":sum(t["steps"] for h in history for t in h["train"].values())},snapshot
