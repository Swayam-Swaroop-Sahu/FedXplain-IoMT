import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.revision_data import assign_split, prepare_clients, read_config
from src.revision_metrics import metrics
from src.revision_probes import similarity
from src.revision_train import average, make_model, proximal_penalty
from src.run_revision import baseline_grid


def rows():
    return pd.DataFrame({"group": np.repeat(np.arange(30),2), "x": np.repeat(np.arange(30),2),
                         "y": np.tile([0,1],30), "record_id":[str(i) for i in range(60)],
                         "client":"bt", "label":np.tile(["Benign","Attack"],30)})


def test_split_and_reproducibility():
    data=assign_split(rows(),42,.2,.2)
    assert data.groupby("group")["split"].nunique().max()==1
    pd.testing.assert_frame_equal(data,assign_split(rows(),42,.2,.2))


def test_official_split_group_conflict():
    data=rows()
    data["official_split"]=np.tile(["train","test"],30)
    with pytest.raises(ValueError,match="crosses"):
        assign_split(data,42,.2,.2)


def test_scaler_fits_train_only():
    data=assign_split(rows(),42,.2,.2)
    train=data.loc[data["split"].eq("train"),"x"].to_numpy()
    test=data.loc[data["split"].eq("test"),"x"].to_numpy()
    fit=prepare_clients(data,["x"])["bt"]
    np.testing.assert_allclose(fit["test"]["x"].ravel(),(test-train.mean())/train.std(),atol=1e-6)


def test_binary_metrics():
    y=np.array([1,1,1,0,0])
    z=np.array([2,1,-1,2,-1])
    out=metrics(y,z)
    pred=expit(z)>=.5
    assert out["balanced_accuracy"]==balanced_accuracy_score(y,pred)
    assert out["macro_f1"]==f1_score(y,pred,average="macro")
    assert out["brier"]==np.mean((expit(z)-y)**2)


def test_similarity_ties_and_zero():
    out=similarity(np.zeros(6),np.arange(6))
    assert out["spearman"] is None and out["cosine"] is None
    assert out["jaccard"]==4/6
    assert similarity(np.arange(6),np.arange(6))["spearman"]==1


def test_average_and_penalty_gradient():
    a,b=torch.nn.Linear(1,1),torch.nn.Linear(1,1)
    with torch.no_grad():
        a.weight.fill_(1)
        b.weight.fill_(5)
    assert average([a,b],[1,3])[0].item()==4
    p=torch.tensor([2.,-1.],requires_grad=True)
    ref=torch.tensor([0.,1.])
    proximal_penalty([p],[ref],.1).backward()
    torch.testing.assert_close(p.grad,.1*(p-ref))


def test_matched_initialization():
    cfg=read_config()
    torch.manual_seed(7)
    a=make_model(cfg)
    torch.manual_seed(7)
    b=make_model(cfg)
    assert all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters()))
    assert ("fedprox",0) in baseline_grid(cfg)
