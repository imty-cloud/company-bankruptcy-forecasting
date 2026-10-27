# -*- coding: utf-8 -*-
"""Tính SHAP (TreeSHAP gốc của XGBoost, log-odds) cho các mô hình XGB: toàn cục + từng công ty ở tập test."""
import sys, json, warnings
sys.path.insert(0, "/home/phantichdulieu/work/repo")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib, xgboost as xgb
import run_study_v2 as R

OUT = "/home/phantichdulieu/work/app/data"; MODELS = "/home/phantichdulieu/work/models"
glob_rows, local = [], {}
for k in R.FILES:
    te = pd.read_csv(f"{OUT}/test_{k}year.csv.gz"); y = te.pop("y_true").values
    for m in ("xgb", "xgb_nan"):
        for mode in ("unweighted", "weighted"):
            obj = joblib.load(f"{MODELS}/{m}_{mode}_{k}year.joblib")
            pipe = obj["pipeline"]
            Xt = R.transform_for_trees(pipe, te)
            booster = pipe.named_steps["clf"].model_.get_booster()
            contrib = booster.predict(xgb.DMatrix(Xt.values, missing=np.nan), pred_contribs=True)
            sv, base = contrib[:, :-1], float(contrib[0, -1])
            cols = list(Xt.columns); ma = np.abs(sv).mean(axis=0); tot = ma.sum()
            for f, v in zip(cols, ma):
                glob_rows.append(dict(file=f"{k}year", model=m, mode=mode, feature=f, family=R.family_of(f),
                                      mean_abs_shap=v, share=v / tot))
            if mode == "unweighted":     # chế độ chính (chốt trước khi xem kết quả)
                top = np.argsort(-np.abs(sv), axis=1)[:, :8]
                rec = []
                for i in range(len(sv)):
                    rec.append([[cols[j], (None if np.isnan(Xt.values[i, j]) else round(float(Xt.values[i, j]), 4)),
                                 round(float(sv[i, j]), 3)] for j in top[i]])
                local[f"{k}|{m}"] = dict(base=round(base, 4), rows=rec)
    print("SHAP", k, flush=True)
g = pd.DataFrame(glob_rows)
g.to_csv(f"{OUT}/shap_global.csv", index=False, float_format="%.5g")
json.dump(local, open("/home/phantichdulieu/work/shap_local.json", "w"), separators=(",", ":"))
# kiểm tra với bang12: top-10 XGB không TS
for k in R.FILES:
    s = g[(g.file == f"{k}year") & (g.model == "xgb") & (g["mode"] == "unweighted")].nlargest(5, "mean_abs_shap")
    print(k, [(r.feature, round(r.mean_abs_shap, 2)) for r in s.itertuples()])
print("DONE")
