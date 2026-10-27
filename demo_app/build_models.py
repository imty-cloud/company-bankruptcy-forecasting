# -*- coding: utf-8 -*-
"""Huấn luyện lại 40 mô hình cuối (5 tệp x 4 mô hình x 2 chế độ TS) từ siêu tham số đã chọn trong báo cáo (bang02),
tái hiện run_study_v2.py (seed 42), rồi xuất dữ liệu cho app demo."""
import sys, os, warnings
sys.path.insert(0, "/home/phantichdulieu/work/repo")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
import run_study_v2 as R
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, ParameterSampler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

DATA = "/home/phantichdulieu/work/repo"; OUT = "/home/phantichdulieu/work/app/data"; MODELS = "/home/phantichdulieu/work/models"
os.makedirs(MODELS, exist_ok=True); os.makedirs(OUT, exist_ok=True)
A = "/mnt/user-data/attachments/38940cab-c1da-5252-8963-781c498fca4a/"
b2 = pd.read_csv(A + "8dffbe2e-bang02_sieu_tham_so.csv", decimal=",", dtype=str)
cfg = dict(add_indicator=True, lr_robust="quantile", dedup=True, search="wide", n_iter=56)
colmap = {"LR không TS": ("lr", False), "LR có TS": ("lr", True), "LR+QT không TS": ("lr_robust", False), "LR+QT có TS": ("lr_robust", True),
          "XGB không TS": ("xgb", False), "XGB có TS": ("xgb", True), "XGB-NaN không TS": ("xgb_nan", False), "XGB-NaN có TS": ("xgb_nan", True)}
KEY = {"d": "clf__max_depth", "η": "clf__learning_rate", "n": "clf__n_estimators", "λ": "clf__reg_lambda",
       "cs": "clf__colsample_bytree", "ss": "clf__subsample", "mcw": "clf__min_child_weight"}


def num(s):
    return float(s.strip().replace(",", "."))


def parse(s):
    a, b = s.split(";")
    ll = num(b.split("=")[1])
    d = {}
    for kv in a.split(", "):
        k_, v = kv.split("=")
        d[k_.strip()] = num(v)
    return d, ll


def best_params(model, weighted, raw):
    d, ll = parse(raw)
    pipe, dist = R.build_pipeline(model, weighted, cfg, 0)
    cands = list(ParameterSampler(dist, n_iter=56, random_state=R.SEED))
    if "C" in d:
        return min(cands, key=lambda c: abs(np.log(c["clf__C"]) - np.log(d["C"]))), ll
    tgt = {KEY[a]: v for a, v in d.items()}
    for c in cands:
        if all(abs(float(c[kk]) - v) < 1e-9 for kk, v in tgt.items()):
            return c, ll
    raise ValueError(f"no cand {raw}")


rows = []
for k in R.FILES:
    Xtr, Xte, ytr, yte = R.load_split(DATA, k, True, R.SEED)
    prior = float(ytr.mean()); rs = R.search_seed(k, R.SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=R.SEED)
    te_df = pd.DataFrame({"y_true": yte}); oof_df = pd.DataFrame({"y_train": ytr})
    Xte.assign(y_true=yte).to_csv(f"{OUT}/test_{k}year.csv.gz", index=False, float_format="%.6g")
    for col, (m, w) in colmap.items():
        raw = b2.loc[b2["Tệp"] == f"{k}year", col].iloc[0]
        params, ll_rep = best_params(m, w, raw)
        pipe, _ = R.build_pipeline(m, w, cfg, rs); pipe.set_params(**params)
        pipe.fit(Xtr, ytr)
        oof = cross_val_predict(clone(pipe), Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        cv_ll = R.log_loss_np(ytr, oof)
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(R._logit(oof), ytr)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, ytr)
        cal = {"platt": [float(platt.coef_[0, 0]), float(platt.intercept_[0])],
               "iso_x": iso.X_thresholds_.astype(float).tolist(), "iso_y": iso.y_thresholds_.astype(float).tolist()}
        p = pipe.predict_proba(Xte)[:, 1]
        pc = {"none": p, "platt": platt.predict_proba(R._logit(p))[:, 1], "isotonic": iso.predict(p)}
        oc = {"none": oof, "platt": platt.predict_proba(R._logit(oof))[:, 1], "isotonic": iso.predict(oof)}
        tag = f"{m}_{'weighted' if w else 'unweighted'}"
        for c in R.CALS:
            te_df[f"{tag}_{c}"] = pc[c].astype(np.float32); oof_df[f"{tag}_{c}"] = oc[c].astype(np.float32)
        tp, fp, fn, tn = R.confusion(yte, pc["none"], R.T_STAR)
        rows.append(dict(file=f"{k}year", model=m, w=w, auc=roc_auc_score(yte, p), ap=average_precision_score(yte, p),
                         brier=brier_score_loss(yte, p), fn=fn, fp=fp, cv_ll=cv_ll, cv_ll_report=ll_rep))
        joblib.dump(dict(pipeline=pipe, cal=cal, prior=prior, params=params, file=k, model=m, weighted=w),
                    f"{MODELS}/{tag}_{k}year.joblib", compress=3)
        print(k, tag, f"AUC={rows[-1]['auc']:.3f} AP={rows[-1]['ap']:.3f} FN/FP={fn}/{fp} cvLL={cv_ll:.3f} (bc {ll_rep})", flush=True)
    te_df.to_csv(f"{OUT}/testpred_{k}year.csv.gz", index=False, float_format="%.6g")
    oof_df.to_csv(f"{OUT}/oofpred_{k}year.csv.gz", index=False, float_format="%.6g")
pd.DataFrame(rows).to_csv("/home/phantichdulieu/work/repro_check.csv", index=False)
print("DONE")
