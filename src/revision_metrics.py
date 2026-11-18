"""Final binary metrics and source-label recall for a binary detector."""
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, log_loss, roc_auc_score


def metrics(y, logits, threshold=0.5, bins=10):
    probability = expit(np.asarray(logits, dtype=float))
    pred = probability >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0,1]).ravel())
    ratio = lambda a, b: a/b if b else None
    specificity, recall = ratio(tn, tn+fp), ratio(tp, tp+fn)
    eligible = len(np.unique(y)) == 2
    reliability = []
    index = np.minimum((probability*bins).astype(int), bins-1)
    for bin_id in np.unique(index):
        mask = index == bin_id
        reliability.append({"bin":int(bin_id),"n":int(mask.sum()),"score":float(probability[mask].mean()),
                            "attack_fraction":float(y[mask].mean())})
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,benign_support=tn+fp,attack_support=tp+fn,
                benign_recall=specificity,attack_recall=recall,fpr=ratio(fp,tn+fp),
                benign_precision=ratio(tn,tn+fn),attack_precision=ratio(tp,tp+fp),
                benign_f1=ratio(2*tn,2*tn+fp+fn),attack_f1=ratio(2*tp,2*tp+fp+fn),
                macro_f1=float(f1_score(y,pred,average="macro",labels=[0,1],zero_division=0)),
                balanced_accuracy=(specificity+recall)/2 if eligible else None,
                roc_auc=float(roc_auc_score(y,logits)) if eligible else None,
                pr_auc=float(average_precision_score(y,logits)) if eligible else None,
                brier=float(np.mean((probability-y)**2)),log_loss=float(log_loss(y,probability,labels=[0,1])),
                ece=float(sum(b["n"]*abs(b["score"]-b["attack_fraction"]) for b in reliability)/len(y)),
                reliability=reliability)


def label_recall(labels, logits, threshold=0.5):
    prediction = expit(logits.astype(float)) >= threshold
    return {str(label):{"support":int(mask.sum()),
                        "recall":float(np.mean(prediction[mask] == (label.casefold() != "benign")))}
            for label in np.unique(labels) if (mask := labels == label).any()}
