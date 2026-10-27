# -*- coding: utf-8 -*-
"""Demo: Dự báo phá sản doanh nghiệp theo nhiều thời hạn (LR, LR+QT, XGB, XGB-NaN).
Chạy:  streamlit run app.py
Mô hình được huấn luyện bằng build_models.py từ siêu tham số trong báo cáo (seed 42)."""
import os, re, sys
import numpy as np, pandas as pd, joblib, xgboost as xgb
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_study_v2 as R          # chứa lớp XGBWeighted/Winsorizer mà mô hình đã lưu cần để nạp lại

st.set_page_config(page_title="Dự báo phá sản doanh nghiệp", page_icon="📉", layout="wide")

MODELS = {"LR": "lr", "LR+QT": "lr_robust", "XGB": "xgb", "XGB-NaN": "xgb_nan"}
MCOL = {"LR": "#7586a6", "LR+QT": "#e8801a", "XGB": "#2c66e0", "XGB-NaN": "#1f9a6c"}
CALS = {"Thô": "none", "Platt": "platt", "Isotonic": "isotonic"}
FAMC = {"Liquidity": "#2c66e0", "Leverage": "#e8801a", "Profitability": "#1f9a6c", "Activity": "#7586a6",
        "Size/Growth": "#9a6bd6", "Other": "#b8a14a", "Missing": "#d1344f"}
HOR = lambda k: f"{k}year (dự báo {6 - k} năm)"
STAR = "t&#42;"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
html, body, .stApp, [data-testid="stMarkdownContainer"], .stMarkdown, label, button, input, select, textarea {font-family:'IBM Plex Sans','Be Vietnam Pro',system-ui,sans-serif !important}
h1,h2,h3,h4 {font-family:'Be Vietnam Pro','IBM Plex Sans',sans-serif !important; letter-spacing:-.01em; color:#12203a}
.block-container {padding-top:1.4rem; padding-bottom:3rem; max-width:1180px}
[data-testid="stHeader"] {background:transparent}
#MainMenu, footer {visibility:hidden}
[data-testid="stSidebar"] {background:#fff; border-right:1px solid #d5dcea}
[data-testid="stSidebar"] .block-container {padding-top:1rem}
[data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border-radius:10px !important; border-color:#d5dcea !important}
.stTabs [data-baseweb="tab-list"] {gap:2px; border-bottom:1px solid #d5dcea}
.stTabs [data-baseweb="tab"] {padding:10px 16px; font-family:'Be Vietnam Pro',sans-serif; font-weight:500; color:#56637f}
.stTabs [aria-selected="true"] {color:#2c66e0 !important}
.hero {display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin-bottom:6px}
.hero h1 {font-size:30px; margin:0; line-height:1.2}
.sub {color:#56637f; margin:2px 0 14px; max-width:78ch}
.pill {display:inline-block; padding:3px 11px; border-radius:99px; font:600 12px 'Be Vietnam Pro',sans-serif; background:#e9eef7; color:#2c66e0; margin-right:6px}
.chip {display:inline-block; padding:3px 11px; border-radius:99px; font:600 12px 'Be Vietnam Pro',sans-serif}
.chip.bad {background:#fbe6ea; color:#c42846} .chip.good {background:#def3ea; color:#16795a} .chip.warn {background:#fbeedb; color:#a85d08} .chip.gray {background:#e9eef7; color:#56637f}
.cardtitle {font:600 15px 'Be Vietnam Pro',sans-serif; margin:0 0 10px}
.muted {color:#56637f; font-size:13px}
.big {font:700 52px/1 'IBM Plex Mono',monospace; letter-spacing:-.03em; color:#12203a}
.vrow {display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap}
.gauge {position:relative; height:14px; border-radius:7px; background:#e9eef7; margin:34px 4px 30px}
.gauge .fill {position:absolute; inset:0 auto 0 0; border-radius:7px}
.gauge .thr {position:absolute; top:-9px; bottom:-9px; width:2px; background:#c9740f}
.gauge .thr b {position:absolute; top:-20px; left:50%; transform:translateX(-50%); font:500 11px 'IBM Plex Mono',monospace; color:#c9740f; white-space:nowrap}
.gauge .tk {position:absolute; top:19px; font:11px 'IBM Plex Mono',monospace; color:#56637f; transform:translateX(-50%)}
.kpis {display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px}
.kpi {background:#f2f5fa; border-radius:8px; padding:10px 12px}
.kpi small {display:block; color:#56637f; font-size:12px}
.kpi b {font:600 20px 'IBM Plex Mono',monospace; color:#12203a}
table.tbl {border-collapse:collapse; width:100%; font-size:14px}
table.tbl th {font:600 11.5px 'Be Vietnam Pro',sans-serif; color:#56637f; text-transform:uppercase; letter-spacing:.05em; text-align:right; padding:7px 10px; border-bottom:1px solid #d5dcea}
table.tbl td {padding:8px 10px; text-align:right; border-bottom:1px solid #eef1f7; font-family:'IBM Plex Mono',monospace}
table.tbl th:first-child, table.tbl td:first-child {text-align:left; font-family:'IBM Plex Sans',sans-serif}
table.tbl td.flag {background:#fbe6ea; color:#c42846; font-weight:600} table.tbl td.ok {color:#56637f}
table.tbl td.sel {outline:2px solid #2c66e0; outline-offset:-2px}
.dot {display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:8px}
.sb {display:grid; grid-template-columns:150px 1fr 56px; gap:10px; align-items:center; padding:5px 0; font-size:13px}
.sb .nm small {display:block; color:#56637f; font-size:11.5px}
.sb .tr {position:relative; height:16px}
.sb .tr:before {content:""; position:absolute; left:50%; top:-3px; bottom:-3px; width:1px; background:#d5dcea}
.sb .bar {position:absolute; top:2px; height:12px; border-radius:2px}
.sb .v {text-align:right; font:12.5px 'IBM Plex Mono',monospace}
.note {border-left:3px solid #c9740f; background:#fbeedb; padding:9px 13px; border-radius:0 6px 6px 0; font-size:13.5px; margin-top:10px}
.rq {background:#fff; border:1px solid #d5dcea; border-radius:10px; padding:14px 16px; height:100%}
.rq b {display:block; font:600 13px 'Be Vietnam Pro',sans-serif; color:#2c66e0; margin-bottom:4px}
.stat {background:#fff; border:1px solid #d5dcea; border-radius:10px; padding:14px 16px}
.stat small {color:#56637f; font-size:12.5px} .stat b {display:block; font:700 28px 'IBM Plex Mono',monospace}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def H(s):
    st.markdown(re.sub(r"\n\s+", "\n", s).strip(), unsafe_allow_html=True)


# ------------------------------------------------------------------ dữ liệu & mô hình
@st.cache_resource
def load_model(m, w, k):
    return joblib.load(os.path.join(HERE, "models", f"{m}_{'weighted' if w else 'unweighted'}_{k}year.joblib"))


@st.cache_data
def load_test(k):
    d = pd.read_csv(os.path.join(HERE, "data", f"test_{k}year.csv.gz"))
    return d.drop(columns="y_true"), d["y_true"].astype(int).values


@st.cache_data
def load_shap_global():
    return pd.read_csv(os.path.join(HERE, "data", "shap_global.csv"))


@st.cache_data
def load_preds(k):
    return pd.read_csv(os.path.join(HERE, "data", f"testpred_{k}year.csv.gz"))


@st.cache_data
def load_report_table(name):
    t = pd.read_csv(os.path.join(HERE, "data", name), dtype=str)
    rows = []
    for _, r in t.iterrows():
        for col in t.columns[2:]:
            m = re.match(r"([\d,]+) \[([\d,]+); ([\d,]+)\]", r[col]); f = lambda x: float(x.replace(",", "."))
            rows.append(dict(k=int(r["Tệp"][0]), cfg=col, v=f(m[1]), lo=f(m[2]), hi=f(m[3])))
    return pd.DataFrame(rows)


def calibrate(obj, p):
    p = np.asarray(p, dtype=float)
    pc = np.clip(p, 1e-6, 1 - 1e-6); lg = np.log(pc / (1 - pc))
    a, b = obj["cal"]["platt"]
    return {"none": p, "platt": 1 / (1 + np.exp(-(a * lg + b))),
            "isotonic": np.interp(p, obj["cal"]["iso_x"], obj["cal"]["iso_y"])}


def predict(m, w, k, X):
    obj = load_model(m, w, k)
    return calibrate(obj, obj["pipeline"].predict_proba(X)[:, 1])


def explain(m, w, k, X):
    """Đóng góp từng đặc trưng (log-odds): TreeSHAP cho XGB, hệ số × giá trị đã chuẩn hóa cho LR."""
    pipe = load_model(m, w, k)["pipeline"]
    if m in ("xgb", "xgb_nan"):
        Xt = R.transform_for_trees(pipe, X)
        contrib = pipe.named_steps["clf"].model_.get_booster().predict(xgb.DMatrix(Xt.values, missing=np.nan), pred_contribs=True)
        return pd.DataFrame({"feature": list(Xt.columns), "value": Xt.values[0], "contrib": contrib[0, :-1]}), float(contrib[0, -1])
    pre = pipe[:-1]; Z = pre.transform(X)
    names = [n.replace("missingindicator_", "miss_") for n in pre.get_feature_names_out()]
    clf = pipe.named_steps["clf"]
    return pd.DataFrame({"feature": names, "value": Z[0], "contrib": clf.coef_[0] * Z[0]}), float(clf.intercept_[0])


def fam(f):
    return "Missing" if f.startswith("miss_") else R.FEATURE_FAMILY[f]


def nice(f):
    return ("Thiếu " + f[5:]) if f.startswith("miss_") else f


def pc(x, d=1):
    return f"{x * 100:.{d}f}%".replace(".", ",")


def fnum(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def sqrt_pos(x):
    return float(np.sqrt(min(max(x, 0), 1)) * 100)


def plot_layout(fig, h=420, **kw):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="#fff", plot_bgcolor="#fff",
                      font=dict(family="IBM Plex Sans, sans-serif", color="#12203a", size=13),
                      legend=dict(orientation="h", y=-0.2), **kw)
    fig.update_xaxes(gridcolor="#e6ebf4", zeroline=False); fig.update_yaxes(gridcolor="#e6ebf4", zeroline=False)
    return fig


# ------------------------------------------------------------------ thanh bên
with st.sidebar:
    H("<div style='font:700 18px \"Be Vietnam Pro\",sans-serif;line-height:1.25'>Dự báo phá sản<br>doanh nghiệp</div>"
      "<div class='muted' style='margin:4px 0 14px'>Polish Companies Bankruptcy · UCI</div>")
    k = st.selectbox("Thời hạn dự báo", [1, 2, 3, 4, 5], format_func=HOR, help="Theo báo cáo: 1year = dự báo 5 năm, 5year = dự báo 1 năm.")
    mname = st.selectbox("Mô hình", list(MODELS), index=2, help="XGB: điền trung vị + chỉ báo thiếu. XGB-NaN: để XGBoost tự xử lý giá trị thiếu. LR+QT: có QuantileTransformer.")
    w = st.radio("Trọng số lớp (TS)", [False, True], format_func=lambda v: "Có TS" if v else "Không TS", horizontal=True)
    cal_name = st.radio("Xác suất", list(CALS), horizontal=True, help="Thô, hoặc đã hiệu chỉnh bằng Platt hay Isotonic (học trên OOF của train).")
    ratio = st.slider("Chi phí bỏ sót : báo nhầm", 1, 30, 10, help="Bỏ sót một công ty phá sản tốn gấp bao nhiêu lần báo nhầm một công ty lành mạnh.")
    t_star = 1 / (1 + ratio)
    H(f"<div class='kpi'><small>Ngưỡng quyết định {STAR} = 1/(1+{ratio})</small><b>{pc(t_star, 2)}</b></div>"
      f"<div class='muted' style='margin-top:8px'>Gắn cờ nếu xác suất ≥ {STAR}. Chỉ đúng khi xác suất đã hiệu chuẩn.</div>")
    if w and cal_name == "Thô" and mname.startswith("LR"):
        H("<div class='note'>LR có TS với xác suất thô bị lệch nặng. Hãy chọn Platt hoặc Isotonic.</div>")
    st.markdown("[Mã nguồn trên GitHub](https://github.com/imty-cloud/company-bankruptcy-forecasting)")

PK = f"{MODELS[mname]}_{'weighted' if w else 'unweighted'}_{CALS[cal_name]}"
P = load_preds(k); y_all = P["y_true"].values; p_cfg = P[PK].values
X, y = load_test(k)

H(f"<div class='hero'><h1>Dự báo phá sản doanh nghiệp theo nhiều thời hạn</h1></div>"
  f"<div class='sub'>So sánh hồi quy logistic và XGBoost, hiệu chỉnh xác suất, giải thích bằng SHAP và quyết định theo chi phí. "
  f"Đang xem <b>{HOR(k)}</b> với <b>{mname}</b>, {'có' if w else 'không'} TS, xác suất {cal_name.lower()}.</div>"
  f"<span class='pill'>5 thời hạn</span><span class='pill'>4 mô hình</span><span class='pill'>64 chỉ số tài chính</span>")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dự báo công ty", "Danh sách cảnh báo", "Chi phí và ngưỡng", "Kết quả nghiên cứu", "Chấm điểm CSV"])

# ================================================================== TAB 1
with tab1:
    n = len(y_all)
    kinds = ["Tất cả công ty", "Thực tế phá sản", "Thực tế không phá sản", "Đang bị gắn cờ", "Phá sản nhưng bị bỏ sót"]
    flagged = p_cfg >= t_star
    pools = {kinds[0]: np.arange(n), kinds[1]: np.where(y_all == 1)[0], kinds[2]: np.where(y_all == 0)[0],
             kinds[3]: np.where(flagged)[0], kinds[4]: np.where((y_all == 1) & ~flagged)[0]}
    st.session_state.setdefault("ma", int(pools[kinds[1]][0]) + 1)
    st.session_state.ma = min(max(int(st.session_state.ma), 1), n)

    def go_to(delta=None, rnd=False):
        pool = pools[st.session_state.get("kind", kinds[1])]
        if len(pool) == 0:
            return
        if rnd:
            st.session_state.ma = int(np.random.choice(pool)) + 1
        else:
            cur = st.session_state.ma - 1
            pos = int(np.searchsorted(pool, cur))
            st.session_state.ma = int(pool[(pos + delta) % len(pool)]) + 1

    left, right = st.columns([1, 2], gap="medium")
    with left:
        with st.container(border=True):
            H("<div class='cardtitle'>Chọn công ty</div>")
            kind = st.selectbox("Nhóm công ty", kinds, index=1, key="kind")
            b1, b2, b3 = st.columns([1, 1.7, 1])
            b1.button("‹", on_click=go_to, kwargs=dict(delta=-1), width="stretch")
            b2.button("Ngẫu nhiên", on_click=go_to, kwargs=dict(rnd=True), width="stretch")
            b3.button("›", on_click=go_to, kwargs=dict(delta=1), width="stretch")
            st.number_input("Mã (số thứ tự trong tập test)", 1, n, key="ma",
                            help="Dữ liệu UCI ẩn danh nên không có mã doanh nghiệp thật. Mã là số thứ tự trong tập test.")
            idx = int(st.session_state.ma) - 1
            H(f"<div class='muted'>{len(pools[kind])} công ty trong nhóm này, tổng {n} công ty trong tập test.</div>")
        with st.container(border=True):
            H("<div class='cardtitle'>Thử thay đổi chỉ số</div>"
              "<div class='muted' style='margin-bottom:6px'>Sửa giá trị, hoặc xóa ô để coi là thiếu dữ liệu. "
              "Thử xóa Attr27 để thấy tín hiệu thiếu đẩy xác suất lên.</div>")
            g = load_shap_global()
            g = g[(g.model == "xgb") & (g["mode"] == "unweighted") & (g.file == f"{k}year")].nlargest(12, "mean_abs_shap")
            keys = list(dict.fromkeys(f.replace("miss_", "") for f in g.feature))[:8]
            base_row = X.iloc[[idx]].copy(); orig_row = base_row.copy()
            edit = pd.DataFrame({"Chỉ số": keys, "Họ": [R.FEATURE_FAMILY[f] for f in keys], "Giá trị": [base_row.iloc[0][f] for f in keys]})
            ed = st.data_editor(edit, hide_index=True, disabled=["Chỉ số", "Họ"], width="stretch", key=f"ed_{k}_{idx}",
                                column_config={"Giá trị": st.column_config.NumberColumn(format="%.4f")})
            for f, v in zip(ed["Chỉ số"], ed["Giá trị"]):
                base_row[f] = np.nan if pd.isna(v) else float(v)
            modified = not np.allclose(base_row.values.astype(float), orig_row.values.astype(float), equal_nan=True)

    with right:
        res = {nm: predict(m, w, k, base_row) for nm, m in MODELS.items()}
        res0 = predict(MODELS[mname], w, k, orig_row)[CALS[cal_name]][0] if modified else None
        p = float(res[mname][CALS[cal_name]][0]); yt = int(y_all[idx]); flag = p >= t_star
        if modified:
            outcome = ("gray", "Đã chỉnh sửa chỉ số")
        elif yt:
            outcome = ("good", "Cảnh báo đúng") if flag else ("bad", "Bỏ sót")
        else:
            outcome = ("warn", "Báo nhầm") if flag else ("good", "Không gắn cờ, đúng")
        fillc = "#d1344f" if flag else "#2c66e0"
        ticks = [0.01, 0.05, 0.1, 0.25, 0.5, 1]
        delta_txt = ""
        if modified:
            d = (p - res0) * 100
            delta_txt = f"<div class='muted' style='margin-top:4px'>Gốc {pc(res0, 2)} → sau chỉnh {pc(p, 2)} ({'+' if d >= 0 else '−'}{fnum(abs(d), 2)} điểm %)</div>"
        H(f"""<div style="background:#fff;border:1px solid #d5dcea;border-radius:10px;padding:18px 20px">
        <div class="vrow"><div class="cardtitle" style="margin:0">Công ty #{idx + 1} · {HOR(k)}</div>
        <span class="chip {'bad' if yt else 'good'}">Thực tế: {'đã phá sản' if yt else 'không phá sản'}</span></div>
        <div class="vrow" style="margin-top:14px;align-items:flex-end">
        <div><div class="muted">Xác suất phá sản ({mname}, {cal_name.lower()})</div><div class="big">{pc(p, 2 if p < 0.1 else 1)}</div>{delta_txt}</div>
        <div style="text-align:right"><span class="chip {outcome[0]}">{outcome[1]}</span>
        <div style="margin-top:8px;font:600 22px 'Be Vietnam Pro',sans-serif;color:{'#c42846' if flag else '#16795a'}">{'GẮN CỜ' if flag else 'Không gắn cờ'}</div>
        <div class="muted">{pc(p, 2)} {'≥' if flag else '&lt;'} {STAR} = {pc(t_star, 2)}</div></div></div>
        <div class="gauge"><div class="fill" style="width:{sqrt_pos(p):.1f}%;background:{fillc}"></div>
        <div class="thr" style="left:{sqrt_pos(t_star):.1f}%"><b>{STAR} {pc(t_star, 1)}</b></div>
        {''.join(f'<span class="tk" style="left:{sqrt_pos(x):.1f}%">{int(x * 100)}%</span>' for x in ticks)}</div>
        <div class="kpis"><div class="kpi"><small>Chi phí kỳ vọng nếu gắn cờ</small><b>{fnum(1 - p, 2)}</b></div>
        <div class="kpi"><small>Chi phí kỳ vọng nếu bỏ qua</small><b>{fnum(p * ratio, 2)}</b></div>
        <div class="kpi"><small>Chọn rẻ hơn</small><b>{'Gắn cờ' if p * ratio > 1 - p else 'Bỏ qua'}</b></div></div>
        <div class="muted" style="margin-top:8px">Đơn vị: 1 = chi phí một lần báo nhầm, bỏ sót tốn {ratio}. Thang đo trên là căn bậc hai của xác suất để thấy rõ vùng nhỏ.</div></div>""")

        with st.container(border=True):
            H("<div class='cardtitle'>Cùng công ty này, bốn mô hình nói gì?</div>")
            head = "".join(f"<th>{c}</th>" for c in CALS)
            body = ""
            for nm in MODELS:
                cells = ""
                for cn, cv in CALS.items():
                    v = float(res[nm][cv][0]); sel = " sel" if nm == mname and cn == cal_name else ""
                    cells += f"<td class='{'flag' if v >= t_star else 'ok'}{sel}'>{pc(v, 2 if v < 0.1 else 1)}</td>"
                body += f"<tr><td><span class='dot' style='background:{MCOL[nm]}'></span>{nm}</td>{cells}</tr>"
            H(f"<table class='tbl'><thead><tr><th>Mô hình ({'có' if w else 'không'} TS)</th>{head}</tr></thead><tbody>{body}</tbody></table>"
              f"<div class='muted' style='margin-top:8px'>Ô đỏ là vượt ngưỡng {STAR}. Ô viền xanh là cấu hình đang chọn ở thanh bên.</div>")

        with st.container(border=True):
            H(f"<div class='cardtitle'>Vì sao {mname} cho kết quả này</div>")
            df, base = explain(MODELS[mname], w, k, base_row)
            pr = float(predict(MODELS[mname], w, k, base_row)["none"][0]); lg = np.log(pr / (1 - pr))
            top = df.reindex(df.contrib.abs().sort_values(ascending=False).index).head(9)
            rest = lg - base - float(top.contrib.sum())
            rows = [(r.feature, r.value, r.contrib) for r in top.itertuples()] + [("(các đặc trưng còn lại)", None, rest)]
            mx = max(abs(c) for _, _, c in rows) or 1
            out = ""
            for f, v, c in rows:
                wd = abs(c) / mx * 50
                if f.startswith("("):
                    nm_, sub = f, ""
                else:
                    vt = ("có" if v == 1 else "không") if f.startswith("miss_") else (f"{v:.3f}".replace(".", ","))
                    nm_, sub = nice(f), f"{fam(f)} · {vt}"
                col = "#d1344f" if c >= 0 else "#1b8f64"; side = "left:50%" if c >= 0 else "right:50%"
                out += (f"<div class='sb'><div class='nm'>{nm_}<small>{sub}</small></div><div class='tr'><div class='bar' style='{side};width:{wd:.1f}%;background:{col}'></div></div>"
                        f"<div class='v'>{'+' if c >= 0 else '−'}{fnum(abs(c), 2)}</div></div>")
            kind_txt = "TreeSHAP gốc của XGBoost" if MODELS[mname] in ("xgb", "xgb_nan") else "hệ số × giá trị đã chuẩn hóa của hồi quy logistic"
            H(out + f"<div class='muted' style='margin-top:8px'>Đơn vị log-odds của mô hình cơ sở chưa hiệu chỉnh ({kind_txt}). Đỏ đẩy lên phía phá sản, xanh kéo xuống. "
                    f"Xác suất thô {pc(pr, 2)}.</div>")

        mtr_auc, mtr_ap, mtr_br = roc_auc_score(y_all, p_cfg), average_precision_score(y_all, p_cfg), brier_score_loss(y_all, p_cfg)
        tp = int(((p_cfg >= t_star) & (y_all == 1)).sum()); fp = int(((p_cfg >= t_star) & (y_all == 0)).sum()); fn = int(((p_cfg < t_star) & (y_all == 1)).sum())
        cost = ratio * fn + fp; base_cost = ratio * int(y_all.sum())
        with st.container(border=True):
            H(f"<div class='cardtitle'>Hiệu năng của cấu hình này trên cả tập test ({n} công ty, {int(y_all.sum())} phá sản)</div>"
              f"<div class='kpis'><div class='kpi'><small>AUC</small><b>{fnum(mtr_auc)}</b></div><div class='kpi'><small>AP</small><b>{fnum(mtr_ap)}</b></div>"
              f"<div class='kpi'><small>Brier</small><b>{fnum(mtr_br, 4)}</b></div>"
              f"<div class='kpi'><small>Recall tại {STAR}</small><b>{fnum(tp / max(tp + fn, 1))}</b></div>"
              f"<div class='kpi'><small>Precision tại {STAR}</small><b>{fnum(tp / max(tp + fp, 1))}</b></div>"
              f"<div class='kpi'><small>Chi phí ({ratio}×FN+FP)</small><b>{cost}</b></div>"
              f"<div class='kpi'><small>So với không gắn cờ ai</small><b>{cost / base_cost * 100:.0f}%</b></div></div>")

# ================================================================== TAB 2
with tab2:
    lst = pd.DataFrame({"Mã": np.arange(n) + 1, "Xác suất phá sản": p_cfg, "Thực tế": np.where(y_all == 1, "Phá sản", "Không phá sản")})
    lst = lst[lst["Xác suất phá sản"] >= t_star].sort_values("Xác suất phá sản", ascending=False)
    lst["Kết quả"] = np.where(lst["Thực tế"] == "Phá sản", "Cảnh báo đúng", "Báo nhầm")
    nflag, ntp = len(lst), int((lst["Thực tế"] == "Phá sản").sum()); nfn = int(((p_cfg < t_star) & (y_all == 1)).sum())
    H(f"<div class='cardtitle'>Công ty vượt ngưỡng {STAR} = {pc(t_star, 2)} ({mname}, {cal_name.lower()}, {'có' if w else 'không'} TS)</div>"
      f"<div class='kpis' style='margin-bottom:12px'><div class='stat'><small>Bị gắn cờ</small><b>{nflag}</b></div>"
      f"<div class='stat'><small>Phá sản thật</small><b style='color:#16795a'>{ntp}</b></div>"
      f"<div class='stat'><small>Báo nhầm</small><b style='color:#a85d08'>{nflag - ntp}</b></div>"
      f"<div class='stat'><small>Bỏ sót</small><b style='color:#c42846'>{nfn}</b></div></div>"
      f"<div class='muted' style='margin-bottom:8px'>Dữ liệu UCI ẩn danh, không có tên hay mã công ty thật; Mã là số thứ tự (từ 1) trong tập test. Nhập mã này ở tab Dự báo công ty để xem chi tiết.</div>")
    st.dataframe(lst, hide_index=True, width="stretch", height=430,
                 column_config={"Xác suất phá sản": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0)})
    st.download_button("Tải danh sách (CSV)", lst.to_csv(index=False), file_name=f"canh_bao_{k}year.csv")

# ================================================================== TAB 3
with tab3:
    st.markdown(f"##### Đường cong chi phí: {ratio} × FN + FP ({mname}, {'có' if w else 'không'} TS)")
    ts = np.round(np.arange(0.01, 0.5001, 0.005), 3)
    fig = go.Figure()
    for (cn, cv), colr in zip(CALS.items(), ["#7586a6", "#e8801a", "#1f9a6c"]):
        p_ = P[f"{MODELS[mname]}_{'weighted' if w else 'unweighted'}_{cv}"].values
        cost_ = np.array([ratio * ((p_ < t) & (y_all == 1)).sum() + ((p_ >= t) & (y_all == 0)).sum() for t in ts])
        fig.add_trace(go.Scatter(x=ts, y=cost_, name=cn, line=dict(color=colr, width=2.4)))
        b = int(np.argmin(cost_)); fig.add_trace(go.Scatter(x=[ts[b]], y=[cost_[b]], mode="markers", showlegend=False,
                                                            marker=dict(color=colr, size=10, line=dict(color="#fff", width=2)),
                                                            hovertemplate=f"{cn}: tốt nhất tại %{{x}}<extra></extra>"))
    fig.add_hline(y=ratio * y_all.sum(), line_dash="dash", line_color="#d1344f", annotation_text="Không gắn cờ ai", annotation_position="top right")
    fig.add_hline(y=(y_all == 0).sum(), line_dash="dot", line_color="#c9740f", annotation_text="Gắn cờ tất cả", annotation_position="bottom right")
    fig.add_vline(x=t_star, line_color="#c9740f", line_width=2, annotation_text="t*")
    plot_layout(fig, 440, xaxis_title="Ngưỡng gắn cờ", yaxis_title="Chi phí tổng")
    st.plotly_chart(fig, width="stretch")
    st.markdown("##### Chi phí tại t* của cả 8 cấu hình (xác suất thô)")
    rows = []
    for nm, m in MODELS.items():
        for ww in (False, True):
            pp = P[f"{m}_{'weighted' if ww else 'unweighted'}_none"].values; pred = pp >= t_star
            tp_, fp_, fn_ = int((pred & (y_all == 1)).sum()), int((pred & (y_all == 0)).sum()), int((~pred & (y_all == 1)).sum())
            c_ = ratio * fn_ + fp_
            rows.append({"Cấu hình": f"{nm} {'có' if ww else 'không'} TS", "Chi phí": c_, "% so với không gắn cờ ai": c_ / (ratio * y_all.sum()),
                         "Recall": tp_ / max(tp_ + fn_, 1), "Precision": tp_ / max(tp_ + fp_, 1), "FN": fn_, "FP": fp_})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"% so với không gắn cờ ai": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=2.1),
                                "Recall": st.column_config.NumberColumn(format="%.3f"), "Precision": st.column_config.NumberColumn(format="%.3f")})
    st.caption("Trên 100% nghĩa là tệ hơn việc không làm gì. LR có TS với xác suất thô gắn cờ gần hết doanh nghiệp, nên chi phí vượt mốc.")

# ================================================================== TAB 4
with tab4:
    H("<div class='kpis' style='margin-bottom:14px'>"
      "<div class='rq'><b>RQ1 · Mô hình</b>XGBoost vẫn tốt hơn, rõ nhất về AP (0,74–0,85). Khoảng cách AUC với LR+QT chỉ 0,008–0,106.</div>"
      "<div class='rq'><b>RQ2 · Hiệu chỉnh</b>Cần cho LR có trọng số lớp, ít tác động với XGBoost.</div>"
      "<div class='rq'><b>RQ3 · SHAP</b>Chỉ báo thiếu, nhất là Attr27, là tín hiệu mạnh. Chưa nên diễn giải như yếu tố kinh tế.</div>"
      "<div class='rq'><b>RQ4 · Chi phí</b>Ở 10:1, XGB không TS chỉ tốn 25–32% mốc không gắn cờ ai. LR có TS thô tốn 119–207%.</div></div>")
    cA, cB = st.columns([3, 2], gap="medium")
    with cA:
        with st.container(border=True):
            metric = st.radio("Chỉ số", ["AUC", "AP"], horizontal=True, index=1, key="metric")
            tw = st.radio("Trọng số lớp", ["Không TS", "Có TS"], horizontal=True, key="tw")
            T = load_report_table("bang03_auc.csv" if metric == "AUC" else "bang04_ap.csv")
            fig = go.Figure()
            for nm in MODELS:
                d = T[T.cfg == f"{nm} {'có TS' if tw == 'Có TS' else 'không TS'}"].sort_values("k")
                fig.add_trace(go.Bar(x=[f"{i}year<br>({6 - i} năm)" for i in d.k], y=d.v, name=nm, marker_color=MCOL[nm],
                                     error_y=dict(type="data", symmetric=False, array=d.hi - d.v, arrayminus=d.v - d.lo, color="#12203a", thickness=1.3, width=3),
                                     hovertemplate="%{y:.3f}<extra>" + nm + "</extra>"))
            plot_layout(fig, 400, barmode="group", yaxis_title=metric, yaxis_range=[0.6 if metric == "AUC" else 0, 1], xaxis_tickangle=0)
            st.plotly_chart(fig, width="stretch")
            st.caption("Xác suất thô, seed 42, khoảng tin cậy 95% bootstrap. Lấy từ Bảng 3 và 4 của báo cáo.")
    with cB:
        with st.container(border=True):
            G = load_shap_global(); G = G[(G.model == "xgb") & (G["mode"] == "unweighted") & (G.file == f"{k}year")].nlargest(10, "mean_abs_shap").iloc[::-1]
            fig = go.Figure(go.Bar(x=G.mean_abs_shap, y=[nice(f) for f in G.feature], orientation="h", marker_color=[FAMC[f] for f in G.family],
                                   customdata=G.family, hovertemplate="%{y} (%{customdata}): %{x:.2f}<extra></extra>"))
            plot_layout(fig, 400, xaxis_title="mean|SHAP| (log-odds)", showlegend=False)
            st.markdown(f"**Top 10 đặc trưng ({HOR(k)}, XGB không TS)**")
            st.plotly_chart(fig, width="stretch")
            H("<div class='muted'>Màu theo họ đặc trưng. Đỏ là chỉ báo thiếu.</div>")

# ================================================================== TAB 5
with tab5:
    H(f"<div class='cardtitle'>Chấm điểm hàng loạt bằng XGB</div><div class='muted' style='margin-bottom:8px'>"
      f"Tải lên CSV có đủ 64 cột Attr1 … Attr64 (ô trống = thiếu). Dùng thiết lập thời hạn, TS, xác suất và tỷ lệ chi phí ở thanh bên.</div>")
    st.download_button("Tải file mẫu (20 công ty)", X.head(20).to_csv(index=False), file_name="mau.csv")
    up = st.file_uploader("File CSV", type="csv")
    if up:
        df = pd.read_csv(up)
        miss = [c for c in R.FEATURES if c not in df.columns]
        if miss:
            st.error(f"Thiếu cột: {', '.join(miss[:8])}…")
        else:
            pr = predict("xgb", w, k, df[R.FEATURES])[CALS[cal_name]]
            out = df.copy(); out["xac_suat_pha_san"] = pr; out["quyet_dinh"] = np.where(pr >= t_star, "Gắn cờ", "Không gắn cờ")
            st.success(f"Đã chấm {len(out)} công ty, {int((pr >= t_star).sum())} công ty bị gắn cờ.")
            st.dataframe(out[["xac_suat_pha_san", "quyet_dinh"]].join(df.iloc[:, :3]), width="stretch",
                         column_config={"xac_suat_pha_san": st.column_config.ProgressColumn("Xác suất phá sản", format="percent", min_value=0.0, max_value=1.0)})
            st.download_button("Tải kết quả", out.to_csv(index=False), file_name="ket_qua.csv")
