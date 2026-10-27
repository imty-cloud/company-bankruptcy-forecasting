# -*- coding: utf-8 -*-
"""
Dự báo phá sản doanh nghiệp theo các thời hạn khác nhau - PHIÊN BẢN 2.
So sánh hiệu năng, hiệu chỉnh xác suất và khả năng giải thích bằng SHAP.
Dữ liệu: Polish Companies Bankruptcy (UCI; Zięba, Tomczak & Tomczak, 2016).

Thay đổi so với run_study.py (mỗi thay đổi bật/tắt bằng cờ dòng lệnh):
  [1] SimpleImputer(median, add_indicator=True) cho LR và XGB  (--no-add-indicator để tắt)
      + biến thể XGB không điền khuyết, để NaN cho XGBoost tự xử lý "xgb_nan"  (--no-xgb-nan)
  [2] Biến thể LR mạnh hơn "lr_robust": QuantileTransformer(normal) hoặc Winsorize 1%/99% trước StandardScaler
      (--lr-robust quantile|winsor|none)
  [3] Loại dòng trùng lặp (đặc trưng + nhãn) trước khi chia train/test  (--no-dedup)
  [4] Tìm kiếm siêu tham số mở rộng: lưới rộng (n_estimators tới 1000) được lấy mẫu MỘT lần với seed cố định
      thành N_CAND=56 tổ hợp; mọi tổ hợp được đánh giá bằng CV 5-fold trên train; random_state mô hình riêng
      cho từng tệp (--search orig để quay về lưới cũ; --n-cand để đổi số tổ hợp)
      Song song hóa ở vòng lặp cấu hình (joblib.Parallel, --n-jobs tiến trình; mỗi cấu hình chạy 1 luồng)
  [5] Lặp lại 5 phép chia 80/20 với seed khác nhau (toàn bộ pipeline: tuning, OOF, hiệu chỉnh, test)
      -> trung bình ± độ lệch chuẩn  (--repeats 1 để tắt)
  [6] Thêm Brier Skill Score (so với dự báo bằng tỷ lệ nền của train), log loss, calibration slope/intercept
  [7] Sửa ECE: bin phân vị không cắt ngang các giá trị trùng (ties) -> không phụ thuộc thứ tự dòng.
      Bản cũ + bootstrap (dương trước, âm sau) làm ECE của isotonic bị thổi phồng có hệ thống
      (--ece-legacy để dùng lại cách cũ; cột ece10_legacy luôn được tính để đối chiếu)
  [8] Cost curve: tổng chi phí theo ngưỡng 0.01-0.5, đánh dấu t* = 1/11  (--no-cost-curve)

Cách chạy:
    python run_study_v2.py --out ./out_v2 --stage all
    python run_study_v2.py --out ./out_v2_smoke --files 1 --smoke      # chạy thử
Các stage: plan (in kế hoạch + ước tính thời gian) | eda | fit | eval | shap | report | all
Giao thức khóa tập test: mọi lựa chọn (siêu tham số, calibrator, ngưỡng P2) chỉ dựa trên tập train
(CV / OOF). Chỉ số trên test được tính đúng MỘT lần cho mỗi phép chia, ở cuối run_pipeline().
"""
import argparse, json, os, sys, time, datetime, platform, warnings
import numpy as np, pandas as pd, joblib
from scipy.io import arff
from scipy.optimize import brentq
from scipy.stats import loguniform, spearmanr
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV, GridSearchCV, ParameterSampler,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer

try:
    from xgboost import XGBClassifier
except ImportError:           # KHÔNG thay thế âm thầm: lỗi sẽ được báo rõ khi cần dùng XGBoost
    XGBClassifier = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning,ignore::UserWarning")   # cho worker joblib

# ----------------------------------------------------------------------------------------------
# 0. CẤU HÌNH
# ----------------------------------------------------------------------------------------------
SEED = 42
FILES = [1, 2, 3, 4, 5]
# GIẢ ĐỊNH THỰC NGHIỆM (header ARFF không chứa metadata để kiểm chứng): file kyear <-> dự báo (6-k) năm
HORIZON_YEARS = {k: 6 - k for k in FILES}
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER_ORIG = 10               # --search orig: RandomizedSearchCV như bản gốc
N_CAND = 56                    # --search wide: số tổ hợp cố định lấy mẫu từ lưới rộng (seed SEED)
N_REPEATS = 5                  # số phép chia 80/20 (seed SEED, SEED+1, ...); seed SEED là phép chia CHÍNH
N_BOOT = 1000
ECE_BINS = (10, 5)             # 10 bin chính, 5 bin nhạy cảm
COST_RATIOS = (10, 5, 20)      # C_FN : C_FP ; 10:1 là phân tích chính
PRIMARY_RATIO = 10
T_STAR = 1.0 / (1.0 + PRIMARY_RATIO)
SHAP_PRIMARY_MODE = "unweighted"   # chốt TRƯỚC khi xem kết quả SHAP; chế độ còn lại là phân tích bổ sung
SHAP_PRIMARY_MODEL = "xgb"         # XGB có điền trung vị + chỉ báo thiếu
ALL_MODELS = ("lr", "lr_robust", "xgb", "xgb_nan")
XGB_MODELS = ("xgb", "xgb_nan")
WEIGHT_MODES = ("unweighted", "weighted")
CALS = ("none", "platt", "isotonic")
FEATURES = [f"Attr{i}" for i in range(1, 65)]

# Định nghĩa Attr (theo ghi nhớ từ mô tả UCI) - [cần kiểm tra với trang UCI trước khi trích dẫn]
ATTR_DEF = {
 1:"net profit / total assets",2:"total liabilities / total assets",3:"working capital / total assets",
 4:"current assets / short-term liabilities",
 5:"[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365",
 6:"retained earnings / total assets",7:"EBIT / total assets",8:"book value of equity / total liabilities",
 9:"sales / total assets",10:"equity / total assets",
 11:"(gross profit + extraordinary items + financial expenses) / total assets",12:"gross profit / short-term liabilities",
 13:"(gross profit + depreciation) / sales",14:"(gross profit + interest) / total assets",
 15:"(total liabilities * 365) / (gross profit + depreciation)",16:"(gross profit + depreciation) / total liabilities",
 17:"total assets / total liabilities",18:"gross profit / total assets",19:"gross profit / sales",
 20:"(inventory * 365) / sales",21:"sales (n) / sales (n-1)",22:"profit on operating activities / total assets",
 23:"net profit / sales",24:"gross profit (in 3 years) / total assets",25:"(equity - share capital) / total assets",
 26:"(net profit + depreciation) / total liabilities",27:"profit on operating activities / financial expenses",
 28:"working capital / fixed assets",29:"logarithm of total assets",30:"(total liabilities - cash) / sales",
 31:"(gross profit + interest) / sales",32:"(current liabilities * 365) / cost of products sold",
 33:"operating expenses / short-term liabilities",34:"operating expenses / total liabilities",
 35:"profit on sales / total assets",36:"total sales / total assets",
 37:"(current assets - inventories) / long-term liabilities",38:"constant capital / total assets",
 39:"profit on sales / sales",40:"(current assets - inventory - receivables) / short-term liabilities",
 41:"total liabilities / ((profit on operating activities + depreciation) * (12/365))",
 42:"profit on operating activities / sales",43:"rotation receivables + inventory turnover in days",
 44:"(receivables * 365) / sales",45:"net profit / inventory",46:"(current assets - inventory) / short-term liabilities",
 47:"(inventory * 365) / cost of products sold",48:"EBITDA (profit on operating activities - depreciation) / total assets",
 49:"EBITDA (profit on operating activities - depreciation) / sales",50:"current assets / total liabilities",
 51:"short-term liabilities / total assets",52:"(short-term liabilities * 365) / cost of products sold",
 53:"equity / fixed assets",54:"constant capital / fixed assets",55:"working capital",
 56:"(sales - cost of products sold) / sales",
 57:"(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)",
 58:"total costs / total sales",59:"long-term liabilities / equity",60:"sales / inventory",61:"sales / receivables",
 62:"(short-term liabilities * 365) / sales",63:"sales / short-term liabilities",64:"sales / fixed assets"}

# Từ điển họ đặc trưng - CỐ ĐỊNH trong mã, thiết lập trước khi xem bất kỳ kết quả SHAP nào.
FAMILY_MAP = {
 "Liquidity":     [3, 4, 28, 33, 37, 40, 46, 50, 57],
 "Leverage":      [2, 8, 10, 12, 15, 16, 17, 25, 26, 27, 30, 34, 38, 51, 53, 54, 59],
 "Profitability": [1, 6, 7, 11, 13, 14, 18, 19, 22, 23, 31, 35, 39, 42, 45, 48, 49, 56, 58],
 "Activity":      [9, 20, 32, 36, 43, 44, 47, 52, 60, 61, 62, 63, 64],
 "Size/Growth":   [21, 29, 55],
 "Other":         [5, 24, 41],   # định nghĩa không điển hình / khó phân loại (chuỗi nhiều năm, đơn vị 'ngày', 'tháng')
}
_all = sorted(i for v in FAMILY_MAP.values() for i in v)
assert _all == list(range(1, 65)), "FAMILY_MAP phải phủ đúng 64 đặc trưng, không trùng lặp"
FEATURE_FAMILY = {f"Attr{i}": fam for fam, ids in FAMILY_MAP.items() for i in ids}
MISSING_FAMILY = "Missing"     # họ riêng cho các cột chỉ báo thiếu miss_AttrX (v2)


def family_of(col):
    return MISSING_FAMILY if col.startswith("miss_") else FEATURE_FAMILY[col]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # thư mục chứa script (chạy từ VS Code ở đâu cũng được)
try:
    sys.stdout.reconfigure(encoding="utf-8")                # hiển thị tiếng Việt đúng trên terminal Windows
except Exception:
    pass


def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------
# 1. DỮ LIỆU
# ----------------------------------------------------------------------------------------------
def load_file(data_dir, k, dedup=False):
    d, _ = arff.loadarff(os.path.join(data_dir, f"{k}year.arff"))
    df = pd.DataFrame(d)
    y = df["class"].apply(lambda v: int(v.decode()) if isinstance(v, bytes) else int(v)).astype(int).values
    X = df.drop(columns="class").replace(b"?", np.nan).astype(float)
    X.columns = FEATURES
    if dedup:                                    # [3] loại dòng trùng (đặc trưng + nhãn), giữ lần xuất hiện đầu
        full = X.copy(); full["class"] = y
        keep = ~full.duplicated().values
        X, y = X[keep], y[keep]
    return X.reset_index(drop=True), y


def load_split(data_dir, k, dedup, split_seed=SEED):
    """Một phép chia phân tầng 80/20 cho mỗi file (split_seed = SEED là phép chia chính)."""
    X, y = load_file(data_dir, k, dedup)
    tr, te = train_test_split(np.arange(len(X)), test_size=TEST_SIZE, stratify=y, random_state=split_seed)
    return X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True), y[tr], y[te]


# ----------------------------------------------------------------------------------------------
# 2. MÔ HÌNH & PIPELINE (mọi bước có trạng thái nằm TRONG Pipeline -> được fit lại ở từng fold)
# ----------------------------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Cắt đuôi theo phân vị q / 1-q của train (bỏ qua NaN; NaN được giữ nguyên)."""
    def __init__(self, q=0.01):
        self.q = q
    def fit(self, X, y=None):
        self.lo_, self.hi_ = np.nanquantile(np.asarray(X, dtype=float), [self.q, 1 - self.q], axis=0)
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


class XGBWeighted(ClassifierMixin, BaseEstimator):
    """Bao XGBClassifier để scale_pos_weight = n_neg/n_pos được tính CHỈ trên fold huấn luyện (trong fit)."""
    def __init__(self, weighted=False, n_estimators=200, max_depth=3, learning_rate=0.1, subsample=1.0,
                 colsample_bytree=1.0, min_child_weight=1, reg_lambda=1.0, random_state=SEED, n_jobs=1):
        self.weighted = weighted; self.n_estimators = n_estimators; self.max_depth = max_depth
        self.learning_rate = learning_rate; self.subsample = subsample; self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight; self.reg_lambda = reg_lambda
        self.random_state = random_state; self.n_jobs = n_jobs
    def fit(self, X, y):
        if XGBClassifier is None:
            raise ImportError("Thiếu thư viện 'xgboost' (pip install xgboost). Không thay thế âm thầm bằng mô hình khác.")
        y = np.asarray(y)
        spw = float((y == 0).sum() / (y == 1).sum()) if self.weighted else 1.0
        self.scale_pos_weight_ = spw
        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate,
            subsample=self.subsample, colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight, reg_lambda=self.reg_lambda,
            scale_pos_weight=spw, eval_metric="logloss", tree_method="hist", missing=np.nan,
            random_state=self.random_state, n_jobs=self.n_jobs, verbosity=0)
        self.model_.fit(np.asarray(X, dtype=float), y)
        self.classes_ = np.array([0, 1])
        return self
    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def build_pipeline(model, weighted, cfg, rs):
    """cfg: add_indicator, lr_robust, search. rs: random_state của mô hình."""
    imputer = lambda: SimpleImputer(strategy="median", add_indicator=cfg["add_indicator"])     # [1]
    wide = cfg["search"] == "wide"
    if model in ("lr", "lr_robust"):
        steps = []
        if model == "lr_robust":                                                                # [2]
            if cfg["lr_robust"] == "quantile":
                steps.append(("robust", QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                                                            random_state=rs)))
            elif cfg["lr_robust"] == "winsor":
                steps.append(("robust", Winsorizer(0.01)))
            else:
                raise ValueError("lr_robust yêu cầu --lr-robust quantile|winsor")
        # biến đổi robust đặt TRƯỚC imputer để các cột chỉ báo 0/1 không bị biến đổi phân vị
        steps += [("imputer", imputer()), ("scaler", StandardScaler()),               # chỉ chuẩn hóa cho LR
                  ("clf", LogisticRegression(class_weight="balanced" if weighted else None,
                                             max_iter=5000, solver="lbfgs"))]       # mặc định: phạt L2
        dist = {"clf__C": loguniform(1e-4, 1e3) if wide else loguniform(1e-3, 1e2)}
    elif model in XGB_MODELS:
        steps = [] if model == "xgb_nan" else [("imputer", imputer())]               # KHÔNG chuẩn hóa cho XGBoost
        steps.append(("clf", XGBWeighted(weighted=weighted, random_state=rs)))
        if wide:                                                                                # [4]
            dist = {"clf__n_estimators": [100, 200, 300, 500, 700, 1000], "clf__max_depth": [2, 3, 4, 5, 6, 8],
                    "clf__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1, 0.2],
                    "clf__subsample": [0.6, 0.7, 0.85, 1.0], "clf__colsample_bytree": [0.4, 0.6, 0.8, 1.0],
                    "clf__min_child_weight": [1, 3, 5, 10], "clf__reg_lambda": [0.5, 1, 5, 10, 20]}
        else:
            dist = {"clf__n_estimators": [100, 200, 300, 400], "clf__max_depth": [2, 3, 4, 5, 6],
                    "clf__learning_rate": [0.03, 0.05, 0.1, 0.2], "clf__subsample": [0.7, 0.85, 1.0],
                    "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__min_child_weight": [1, 3, 5],
                    "clf__reg_lambda": [1, 5, 10]}
    else:
        raise ValueError(model)
    return Pipeline(steps), dist


def transform_for_trees(pipe, X):
    """Đầu vào thực tế của cây XGBoost (sau imputer nếu có) dưới dạng DataFrame có tên cột."""
    if len(pipe.steps) == 1:
        return X.copy()
    pre = pipe[:-1]
    names = [n.replace("missingindicator_", "miss_") for n in pre.get_feature_names_out()]
    return pd.DataFrame(pre.transform(X), columns=names)


# ----------------------------------------------------------------------------------------------
# 3. CHỈ SỐ
# ----------------------------------------------------------------------------------------------
LEGACY_ECE = False


def ece_quantile_legacy(y, p, n_bins):
    """ECE bản gốc: cắt thứ tự đã sắp thành n_bins phần bằng nhau -> có thể cắt ngang ties (phụ thuộc thứ tự dòng)."""
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    order = np.argsort(p, kind="mergesort")
    tot = 0.0
    for b in np.array_split(order, n_bins):
        if len(b):
            tot += len(b) / n * abs(y[b].mean() - p[b].mean())
    return tot


def ece_quantile(y, p, n_bins):
    """[7] ECE với bin phân vị; các giá trị p bằng nhau luôn chung một bin (như calibration_curve 'quantile').
    sum_b (n_b/n)*|mean(y)-mean(p)|. Không phụ thuộc thứ tự dòng."""
    if LEGACY_ECE:
        return ece_quantile_legacy(y, p, n_bins)
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))[1:-1]
    b = np.searchsorted(edges, p, side="right")
    cnt = np.bincount(b, minlength=n_bins)
    sy = np.bincount(b, weights=y, minlength=n_bins); sp = np.bincount(b, weights=p, minlength=n_bins)
    nz = cnt > 0
    return float(np.sum(np.abs(sy[nz] - sp[nz])) / n)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def log_loss_np(y, p, eps=1e-15):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_skill(y, p, prior):
    """[6] BSS = 1 - Brier / Brier_ref, Brier_ref = Brier của dự báo hằng bằng tỷ lệ nền TRAIN."""
    ref = float(np.mean((y - prior) ** 2))
    return 1.0 - brier_score_loss(y, p) / ref


def calibration_slope_intercept(y, p):
    """[6] slope: hệ số b trong logit P(y=1) = a + b*logit(p) (lý tưởng 1);
    intercept: calibration-in-the-large, a trong logit P(y=1) = a + 1*logit(p) (offset; lý tưởng 0)."""
    lp = _logit(p).ravel(); y = np.asarray(y)
    slope = float(LogisticRegression(C=1e10, max_iter=1000).fit(lp.reshape(-1, 1), y).coef_[0, 0])
    f = lambda a: np.sum(y - 1.0 / (1.0 + np.exp(-(a + lp))))
    try:
        intercept = float(brentq(f, -30, 30))
    except ValueError:
        intercept = np.nan
    return slope, intercept


def all_metrics(y, p, prior, with_slope=True):
    p = np.asarray(p, dtype=np.float64)          # XGB/isotonic trả về float32: clip 1-1e-15 sẽ thành 1.0
    m = dict(auc=roc_auc_score(y, p),
             ap=average_precision_score(y, p),     # AP = sum_n (R_n - R_{n-1}) P_n (KHÔNG nội suy hình thang)
             brier=brier_score_loss(y, p), bss=brier_skill(y, p, prior), logloss=log_loss_np(y, p),
             ece10=ece_quantile(y, p, 10), ece5=ece_quantile(y, p, 5))
    if with_slope:
        m["cal_slope"], m["cal_intercept"] = calibration_slope_intercept(y, p)
    return m


def confusion(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return tp, fp, fn, tn


def cost_at(y, p, t, cfn, cfp=1.0):
    _, fp, fn, _ = confusion(y, p, t)
    return cfn * fn + cfp * fp


def best_threshold(y, p, cfn, cfp=1.0):
    """Ngưỡng tối thiểu hóa chi phí trên (y, p) (dùng OOF của train). Quy tắc: dự báo phá sản nếu p >= t."""
    y = np.asarray(y); p = np.asarray(p)
    order = np.argsort(p, kind="mergesort"); ps = p[order]; ys = y[order]
    cum_pos = np.concatenate([[0], np.cumsum(ys)]); cum_neg = np.concatenate([[0], np.cumsum(1 - ys)])
    uniq, first = np.unique(ps, return_index=True)
    n_neg = int((y == 0).sum()); n_pos = int((y == 1).sum())
    fn = cum_pos[first]; fp = n_neg - cum_neg[first]
    costs = np.concatenate([cfn * fn + cfp * fp, [cfn * n_pos]])          # ứng viên cuối: không gắn cờ ai
    thr = np.concatenate([uniq, [1.01]])
    return float(thr[int(np.argmin(costs))])


# ----------------------------------------------------------------------------------------------
# 4. HÀM TRUNG TÂM: (file, model, class_weight, split_seed) -> OOF, dự báo test, chỉ số
# ----------------------------------------------------------------------------------------------
def search_seed(k, split_seed):
    """[4] random_state của mô hình: khác nhau theo tệp (và theo phép chia)."""
    return SEED + 100 * k + (split_seed - SEED)


def candidate_grid(dist, n_cand):
    """[4] Tập tổ hợp siêu tham số CỐ ĐỊNH: lấy mẫu n_cand tổ hợp từ lưới rộng với seed SEED
    (giống nhau cho mọi tệp/phép chia/trọng số), dạng param_grid cho GridSearchCV."""
    return [{kk: [v] for kk, v in c.items()} for c in ParameterSampler(dist, n_iter=n_cand, random_state=SEED)]


def run_pipeline(data_dir, k, model, weighted, cfg, split_seed=SEED, n_jobs=1):
    Xtr, Xte, ytr, yte = load_split(data_dir, k, cfg["dedup"], split_seed)
    rs = search_seed(k, split_seed) if cfg["search"] == "wide" else SEED
    pipe, dist = build_pipeline(model, weighted, cfg, rs)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=split_seed)
    prior = float(ytr.mean())
    t0 = time.time()

    # (b1) Tuning: 5-fold CV, tối ưu log loss, CHỈ trên train
    if cfg["search"] == "wide":
        search = GridSearchCV(pipe, candidate_grid(dist, cfg["n_iter"]), cv=cv, scoring="neg_log_loss",
                              n_jobs=n_jobs, refit=True, error_score="raise")
    else:
        search = RandomizedSearchCV(pipe, dist, n_iter=cfg["n_iter"], cv=cv, scoring="neg_log_loss",
                                    random_state=rs, n_jobs=n_jobs, refit=True, error_score="raise")
    search.fit(Xtr, ytr)
    best = search.best_estimator_                 # đã refit trên TOÀN BỘ train

    # (b2) OOF trên train với cấu hình đã chốt (pipeline nội bộ được fit lại ở mỗi fold)
    oof = cross_val_predict(clone(best), Xtr, ytr, cv=cv, method="predict_proba", n_jobs=n_jobs)[:, 1]

    # (c1) Calibrator fit trực tiếp trên OOF
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(oof), ytr)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, ytr)
    oof_cal = {"none": oof, "platt": platt.predict_proba(_logit(oof))[:, 1], "isotonic": iso.predict(oof)}

    # (c2) Dự báo test bằng mô hình refit + calibrator
    p_raw = best.predict_proba(Xte)[:, 1]
    test_cal = {"none": p_raw, "platt": platt.predict_proba(_logit(p_raw))[:, 1], "isotonic": iso.predict(p_raw)}

    # Ngưỡng P2 (chọn trên OOF của train, theo từng cấu hình & từng tỷ lệ chi phí)
    p2 = {(c, r): best_threshold(ytr, oof_cal[c], r) for c in CALS for r in COST_RATIOS}

    # (c3) ĐÁNH GIÁ TEST - đúng MỘT lần, sau khi mọi lựa chọn đã chốt
    rows = []
    for c in CALS:
        m = all_metrics(yte, test_cal[c], prior)
        tp, fp, fn, tn = confusion(yte, test_cal[c], T_STAR)
        rows.append(dict(file=f"{k}year", horizon_years=HORIZON_YEARS[k], split_seed=split_seed, model=model,
                         weighting="weighted" if weighted else "unweighted", calibration=c,
                         n_test=len(yte), n_pos_test=int(yte.sum()), prevalence_test=float(yte.mean()),
                         prevalence_train=prior, **m, thr_t_star=T_STAR, tp=tp, fp=fp, fn=fn, tn=tn,
                         recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                         cost_p1=cost_at(yte, test_cal[c], T_STAR, PRIMARY_RATIO),
                         cost_p2=cost_at(yte, test_cal[c], p2[(c, PRIMARY_RATIO)], PRIMARY_RATIO)))
    return dict(file=k, model=model, weighted=weighted, split_seed=split_seed, cfg_sig=cfg_signature(cfg),
                best_params=search.best_params_, cv_logloss=-search.best_score_, prior=prior,
                n_candidates=len(search.cv_results_["params"]), fit_seconds=time.time() - t0,
                ytr=ytr, yte=yte, oof=oof_cal, test=test_cal, p2=p2, metrics=pd.DataFrame(rows), pipeline=best)


# ----------------------------------------------------------------------------------------------
# 5. STAGES
# ----------------------------------------------------------------------------------------------
def make_cfg(args):
    return dict(add_indicator=args.add_indicator, lr_robust=args.lr_robust, dedup=args.dedup,
                search=args.search, n_iter=(2 if args.smoke else (args.n_cand if args.search == "wide" else N_ITER_ORIG)))


def cfg_signature(cfg):
    return json.dumps(cfg, sort_keys=True)


def split_seeds(args):
    return [SEED + r for r in range(args.repeats)]


def wlabel(w):
    return "weighted" if w else "unweighted"


def write_config(out, args):
    cfg = dict(seed=SEED, files=args.files, horizon_mapping_ASSUMPTION={f"{k}year": f"{HORIZON_YEARS[k]}-year forecast" for k in args.files},
               test_size=TEST_SIZE, cv_folds=CV_FOLDS, split_seeds=split_seeds(args), n_boot=args.n_boot,
               ece_bins=ECE_BINS, ece_legacy=args.ece_legacy, cost_ratios=COST_RATIOS,
               shap_primary_mode=SHAP_PRIMARY_MODE, shap_primary_model=SHAP_PRIMARY_MODEL,
               models=args.models, repeat_models=args.repeat_models, smoke=args.smoke, cost_curve=args.cost_curve, **make_cfg(args),
               python=sys.version, platform=platform.platform(), started=datetime.datetime.now().isoformat())
    import sklearn, scipy
    cfg.update(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
               xgboost=getattr(sys.modules.get("xgboost"), "__version__", None))
    try:
        import shap; cfg["shap"] = shap.__version__
    except ImportError:
        cfg["shap"] = None
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1, default=str)


def stage_eda(args):
    rows = []
    for k in args.files:
        X0, y0 = load_file(args.data_dir, k, dedup=False)
        X, y = load_file(args.data_dir, k, dedup=args.dedup)
        full0 = X0.copy(); full0["class"] = y0
        tr, te = train_test_split(np.arange(len(X)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
        htr = pd.util.hash_pandas_object(X.iloc[tr], index=False); hte = pd.util.hash_pandas_object(X.iloc[te], index=False)
        rows.append(dict(file=f"{k}year", horizon_years_assumed=HORIZON_YEARS[k], n_raw=len(X0),
                         dup_rows_features=int(X0.duplicated().sum()), dup_rows_with_label=int(full0.duplicated().sum()),
                         dedup_applied=args.dedup, n=len(X), n_pos=int(y.sum()),
                         n_neg=int((y == 0).sum()), prevalence_pct=100 * y.mean(),
                         missing_cells_pct=100 * X.isna().values.mean(), rows_any_missing_pct=100 * X.isna().any(axis=1).mean(),
                         max_col_missing_pct=100 * X.isna().mean().max(), max_missing_col=X.isna().mean().idxmax(),
                         attr27_missing_pct=100 * X["Attr27"].isna().mean(),
                         attr27_missing_pct_pos=100 * X["Attr27"][y == 1].isna().mean(),
                         attr27_missing_pct_neg=100 * X["Attr27"][y == 0].isna().mean(),
                         n_train=len(tr), n_test=len(te), pos_train=int(y[tr].sum()), pos_test=int(y[te].sum()),
                         test_rows_duplicated_in_train=int(hte.isin(set(htr)).sum())))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "eda_summary.csv"), index=False)
    fm = pd.DataFrame([dict(feature=f, family=FEATURE_FAMILY[f], definition_UCI_unverified=ATTR_DEF[int(f[4:])])
                       for f in FEATURES])
    fm.to_csv(os.path.join(args.out, "feature_family_map.csv"), index=False)
    fm.groupby("family").size().rename("n_features").to_csv(os.path.join(args.out, "feature_family_counts.csv"))
    log("EDA xong: eda_summary.csv, feature_family_map.csv")


def cache_path(out, k, model, weighted, seed=SEED):
    return os.path.join(out, "cache", f"{k}year_{model}_{'w' if weighted else 'u'}_s{seed}.joblib")


def load_cached(out, k, model, weighted, seed=SEED):
    cp = cache_path(out, k, model, weighted, seed)
    return joblib.load(cp) if os.path.exists(cp) else None


def n_workers(args):
    return os.cpu_count() if args.n_jobs < 0 else max(1, args.n_jobs)


def fit_jobs(args):
    """Danh sách cấu hình (seed, tệp, mô hình, trọng số); XGB và tệp lớn xếp trước để cân tải giữa các tiến trình."""
    jobs = [(s, k, m, w) for s in split_seeds(args) for k in args.files for m in args.models for w in (False, True)
            if s == SEED or m in args.repeat_models]
    size ={k: os.path.getsize(os.path.join(args.data_dir, f"{k}year.arff")) for k in args.files}
    return sorted(jobs, key=lambda j: (j[2] not in XGB_MODELS, -size[j[1]], not j[3]))


def _fit_job(data_dir, out, k, m, w, s, cfg, legacy):
    """Chạy MỘT cấu hình trong tiến trình con (1 luồng) và ghi cache; trả về dòng log."""
    global LEGACY_ECE
    LEGACY_ECE = legacy
    warnings.filterwarnings("ignore")
    res = run_pipeline(data_dir, k, m, w, cfg, split_seed=s, n_jobs=1)
    joblib.dump(res, cache_path(out, k, m, w, s))
    if s == SEED:
        joblib.dump(res["pipeline"], os.path.join(out, "models", f"{m}_{k}year_{'w' if w else 'u'}.joblib"))
    auc = res["metrics"].set_index("calibration").loc["none", "auc"]
    return (f"fit xong seed={s} {k}year {m} {wlabel(w)} ({res['fit_seconds']:.0f}s, {res['n_candidates']} tổ hợp) "
            f"AUC test={auc:.4f} best={res['best_params']}")


def stage_plan(args):
    """In số cấu hình thực tế, số tiến trình song song và ước tính thời gian (benchmark nhỏ, 1 luồng)."""
    cfg = make_cfg(args); jobs = fit_jobs(args); nw = n_workers(args)
    fits_per_job = cfg["n_iter"] * CV_FOLDS + 1 + CV_FOLDS          # tuning + refit + OOF
    log(f"Số tổ hợp siêu tham số mỗi cấu hình: {cfg['n_iter']} ({'cố định, seed ' + str(SEED) if cfg['search'] == 'wide' else 'RandomizedSearchCV'})")
    log(f"Số cấu hình (seed × tệp × mô hình × trọng số): {len(jobs)} = {args.repeats} × {len(args.files)} × "
        f"{len(args.models)} × 2;  số lần fit mỗi cấu hình: {fits_per_job};  tổng số lần fit: {len(jobs) * fits_per_job}")
    log(f"Số tiến trình song song: {nw} (mỗi tiến trình 1 luồng; CPU logic: {os.cpu_count()})")
    # benchmark: mỗi mô hình, vài tổ hợp đầu của lưới cố định, fit trên 80% train của tệp lớn nhất
    kref = max(args.files, key=lambda k: os.path.getsize(os.path.join(args.data_dir, f"{k}year.arff")))
    Xtr, _, ytr, _ = load_split(args.data_dir, kref, cfg["dedup"])
    tr, _ = next(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED).split(Xtr, ytr))
    per_fit = {}
    for m in args.models:
        for w in (False, True):
            pipe, dist = build_pipeline(m, w, cfg, SEED)
            cands = list(ParameterSampler(dist, n_iter=cfg["n_iter"], random_state=SEED))
            sample = cands[:: max(1, len(cands) // 6)][:6]
            t0 = time.time()
            for c in sample:
                clone(pipe).set_params(**c).fit(Xtr.iloc[tr], ytr[tr])
            per_fit[(m, w)] = (time.time() - t0) / len(sample)
    sizes = {k: len(load_file(args.data_dir, k, cfg["dedup"])[1]) for k in args.files}
    cpu = sum(per_fit[(m, w)] * sizes[k] / sizes[kref] * fits_per_job for (s, k, m, w) in jobs)
    for (m, w), t in per_fit.items():
        log(f"  benchmark {m} {wlabel(w)}: {t:.2f} s/lần fit (tệp {kref}year)")
    log(f"Ước tính: {cpu / 3600:.1f} giờ-CPU -> khoảng {cpu / nw / 60 * 1.15:.0f} phút cho stage fit với {nw} tiến trình "
        f"(+ eval/shap/report khoảng 5-10 phút). Benchmark bị chậm hơn nếu máy đang bận.")


def stage_fit(args):
    os.makedirs(os.path.join(args.out, "cache"), exist_ok=True); os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
    cfg = make_cfg(args); sig = cfg_signature(cfg)
    todo = []
    for (s, k, m, w) in fit_jobs(args):
        cp = cache_path(args.out, k, m, w, s)
        if os.path.exists(cp) and not args.force:
            try:
                if joblib.load(cp).get("cfg_sig") == sig:
                    continue
            except Exception:                     # cache hỏng (vd. tiến trình bị dừng khi đang ghi) -> fit lại
                log(f"cache hỏng, fit lại: {os.path.basename(cp)}")
        todo.append((s, k, m, w))
    nw = n_workers(args)
    log(f"fit: {len(todo)} cấu hình cần chạy (bỏ qua {len(fit_jobs(args)) - len(todo)} đã có cache), "
        f"{cfg['n_iter']} tổ hợp siêu tham số mỗi cấu hình, {nw} tiến trình song song")
    t_all = time.time()
    results = Parallel(n_jobs=nw, return_as="generator_unordered")(
        delayed(_fit_job)(args.data_dir, args.out, k, m, w, s, cfg, args.ece_legacy) for (s, k, m, w) in todo)
    for i, msg in enumerate(results, 1):
        log(f"[{i}/{len(todo)}] {msg} (tổng {(time.time()-t_all)/60:.1f} phút)")
    # gộp bảng chỉ số điểm (phép chia chính), tham số tốt nhất, ngưỡng P2, dự báo
    mets, bps, p2s, reps = [], [], [], []
    for k in args.files:
        pred_te = {"y_true": None}; pred_tr = {"y_train": None}
        for m in args.models:
            for w in (False, True):
                for s in split_seeds(args):
                    if s != SEED and m not in args.repeat_models: continue
                    r = load_cached(args.out, k, m, w, s)
                    if r is None: continue
                    reps.append(r["metrics"])
                    if s != SEED: continue
                    mets.append(r["metrics"])
                    bps.append(dict(file=f"{k}year", model=m, weighting=wlabel(w),
                                    cv_logloss=r["cv_logloss"], **{kk: str(vv) for kk, vv in r["best_params"].items()}))
                    for (c, ratio), t in r["p2"].items():
                        p2s.append(dict(file=f"{k}year", model=m, weighting=wlabel(w),
                                        calibration=c, ratio=ratio, thr_p2=t))
                    if pred_te["y_true"] is None:
                        pred_te["y_true"] = r["yte"]; pred_tr["y_train"] = r["ytr"]
                    for c in CALS:
                        col = f"{m}_{wlabel(w)}_{c}"
                        pred_te[col] = r["test"][c]; pred_tr[col] = r["oof"][c]
        if pred_te["y_true"] is not None:
            pd.DataFrame(pred_te).to_csv(os.path.join(args.out, f"test_predictions_{k}year.csv"), index=False)
            pd.DataFrame(pred_tr).to_csv(os.path.join(args.out, f"oof_predictions_{k}year.csv"), index=False)
    if mets:
        pd.concat(mets).to_csv(os.path.join(args.out, "metrics_point.csv"), index=False)
        pd.DataFrame(bps).to_csv(os.path.join(args.out, "best_params.csv"), index=False)
        pd.DataFrame(p2s).to_csv(os.path.join(args.out, "p2_thresholds.csv"), index=False)
    if reps:                                                                                    # [5]
        rp = pd.concat(reps)
        rp.to_csv(os.path.join(args.out, "repeated_split_metrics.csv"), index=False)
        mcols = ["auc", "ap", "brier", "bss", "logloss", "ece10", "ece5", "cal_slope", "cal_intercept", "cost_p1"]
        g = rp.groupby(["file", "model", "weighting", "calibration"])[mcols]
        summ = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).join(g.size().rename("n_splits"))
        summ.reset_index().to_csv(os.path.join(args.out, "repeated_split_summary.csv"), index=False)
        # khoảng cách AUC/AP ghép cặp theo phép chia (cùng tập test)
        gaps = []
        piv = rp[rp.calibration == "none"].set_index(["file", "split_seed", "weighting", "model"])[["auc", "ap"]]
        for (a, b) in GAP_PAIRS:
            if a not in args.models or b not in args.models: continue
            d = (piv.xs(a, level="model") - piv.xs(b, level="model")).dropna().reset_index()
            d["A"], d["B"] = a, b
            gaps.append(d)
        if gaps:
            gp = pd.concat(gaps).rename(columns={"auc": "auc_diff", "ap": "ap_diff"})
            gp.to_csv(os.path.join(args.out, "repeated_split_gaps.csv"), index=False)
            gs = gp.groupby(["file", "weighting", "A", "B"])[["auc_diff", "ap_diff"]].agg(["mean", "std", "count"])
            gs.columns = ["_".join(c) for c in gs.columns]
            gs.reset_index().to_csv(os.path.join(args.out, "repeated_split_gap_summary.csv"), index=False)


GAP_PAIRS = [("xgb", "lr"), ("xgb", "lr_robust"), ("xgb_nan", "lr_robust"), ("lr_robust", "lr"), ("xgb_nan", "xgb")]


def boot_indices(yte, k, n_boot):
    """Sinh chỉ số bootstrap phân tầng MỘT lần cho mỗi file (giữ nguyên số mẫu dương/âm).
    Thứ tự (dương trước, âm sau) giữ như bản gốc để cột ece10_legacy tái lập đúng lỗi của bản gốc;
    các chỉ số còn lại (kể cả ECE mới) không phụ thuộc thứ tự."""
    rng = np.random.default_rng(SEED + k)
    pos = np.where(yte == 1)[0]; neg = np.where(yte == 0)[0]
    return np.stack([np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
                     for _ in range(n_boot)])


BOOT_COLS = ["auc", "ap", "brier", "bss", "logloss", "ece10", "ece5", "ece10_legacy", "cost_p1", "cost_p2"]
LOWER_IS_BETTER = {"brier", "logloss", "ece10", "ece5", "ece10_legacy", "cost_p1", "cost_p2"}


def _boot_one(y, p, idx, t1, t2, cfn, prior, legacy):
    global LEGACY_ECE
    LEGACY_ECE = legacy
    warnings.filterwarnings("ignore")
    out = np.empty((len(idx), len(BOOT_COLS)))
    for b, ii in enumerate(idx):
        yb, pb = y[ii], p[ii]
        m = all_metrics(yb, pb, prior, with_slope=False)
        out[b] = [m["auc"], m["ap"], m["brier"], m["bss"], m["logloss"], m["ece10"], m["ece5"],
                  ece_quantile_legacy(yb, pb, 10), cost_at(yb, pb, t1, cfn), cost_at(yb, pb, t2, cfn)]
    return out


def stage_eval(args):
    cost_rows, bench_rows, ci_rows, diff_rows = [], [], [], []
    for k in args.files:
        yte = None; preds = {}; p2 = {}; prior = None
        for m in args.models:
            for w in (False, True):
                r = load_cached(args.out, k, m, w)
                if r is None: continue
                yte = r["yte"]; prior = r["prior"]; wn = wlabel(w)
                for c in CALS:
                    preds[(m, wn, c)] = r["test"][c]
                    for ratio in COST_RATIOS:
                        p2[(m, wn, c, ratio)] = r["p2"][(c, ratio)]
        if yte is None: continue
        n, npos = len(yte), int(yte.sum()); nneg = n - npos
        # chi phí điểm: P1, P2, benchmark
        for (m, wn, c), p in preds.items():
            for ratio in COST_RATIOS:
                t_star = 1.0 / (1.0 + ratio); t2 = p2[(m, wn, c, ratio)]
                for pol, t in (("P1", t_star), ("P2", t2)):
                    cst = cost_at(yte, p, t, ratio); tp, fp, fn, tn = confusion(yte, p, t)
                    cost_rows.append(dict(file=f"{k}year", model=m, weighting=wn, calibration=c, ratio=f"{ratio}:1",
                                          policy=pol, threshold=t, tp=tp, fp=fp, fn=fn, tn=tn, total_cost=cst,
                                          cost_per_application=cst / n, regret_vs_oracle=cst,
                                          regret_norm_by_approve_all=cst / (ratio * npos)))
        for ratio in COST_RATIOS:
            for name, cst in (("approve_all", ratio * npos), ("reject_all", 1.0 * nneg), ("oracle", 0.0)):
                bench_rows.append(dict(file=f"{k}year", ratio=f"{ratio}:1", benchmark=name, total_cost=cst,
                                       cost_per_application=cst / n, n_test=n, n_pos=npos, n_neg=nneg))
        # bootstrap (chỉ số bootstrap sinh một lần, dùng chung cho mọi mô hình/calibrator/chính sách)
        idx = boot_indices(yte, k, args.n_boot)
        keys = list(preds.keys())
        res = Parallel(n_jobs=args.n_jobs)(delayed(_boot_one)(yte, preds[kk], idx, T_STAR,
                                                              p2[(kk[0], kk[1], kk[2], PRIMARY_RATIO)], PRIMARY_RATIO,
                                                              prior, args.ece_legacy)
                                           for kk in keys)
        boot = {kk: pd.DataFrame(a, columns=BOOT_COLS) for kk, a in zip(keys, res)}
        point = {}
        for kk in keys:
            pt = all_metrics(yte, preds[kk], prior, with_slope=False)
            pt["ece10_legacy"] = ece_quantile_legacy(yte, preds[kk], 10)
            pt["cost_p1"] = cost_at(yte, preds[kk], T_STAR, PRIMARY_RATIO)
            pt["cost_p2"] = cost_at(yte, preds[kk], p2[(kk[0], kk[1], kk[2], PRIMARY_RATIO)], PRIMARY_RATIO)
            point[kk] = pt
        for kk, df in boot.items():
            for col in BOOT_COLS:
                lo, hi = np.percentile(df[col], [2.5, 97.5])
                ci_rows.append(dict(file=f"{k}year", model=kk[0], weighting=kk[1], calibration=kk[2], metric=col,
                                    point=point[kk][col], ci_low=lo, ci_high=hi))

        def add_diff(question, a, b, cols):
            if a not in boot or b not in boot: return
            for col in cols:
                d = boot[a][col].values - boot[b][col].values
                lo, hi = np.percentile(d, [2.5, 97.5])
                pdiff = point[a][col] - point[b][col]
                sig = not (lo <= 0 <= hi)
                if not sig:
                    verdict = "no_diff"
                else:
                    a_better = (hi < 0) if col in LOWER_IS_BETTER else (lo > 0)
                    verdict = "A_better" if a_better else "A_worse"
                diff_rows.append(dict(file=f"{k}year", question=question, metric=col, A="|".join(a), B="|".join(b),
                                      point_diff_A_minus_B=pdiff, boot_mean_diff=d.mean(), ci_low=lo, ci_high=hi,
                                      contains_zero=not sig, verdict=verdict,
                                      sign_mismatch=bool((pdiff < 0 and lo > 0) or (pdiff > 0 and hi < 0))))
        for wn in WEIGHT_MODES:
            for c in CALS:
                for (a, b) in GAP_PAIRS:
                    add_diff(f"RQ1: {a} - {b}", (a, wn, c), (b, wn, c), ["auc", "ap"])
        for m in args.models:
            for wn in WEIGHT_MODES:
                for c in ("platt", "isotonic"):
                    add_diff("RQ2: calibrated - raw", (m, wn, c), (m, wn, "none"),
                             ["brier", "bss", "logloss", "ece10", "ece5", "ece10_legacy"])
                    add_diff("RQ4: cost calibrated - raw (P1, ratio 10:1)", (m, wn, c), (m, wn, "none"), ["cost_p1"])
                    add_diff("RQ4 (supp.): cost calibrated - raw (P2, ratio 10:1)", (m, wn, c), (m, wn, "none"), ["cost_p2"])
            for c in CALS:
                add_diff("RQ2: weighted - unweighted", (m, "weighted", c), (m, "unweighted", c),
                         ["brier", "bss", "ece10", "ece5", "auc", "ap"])
        log(f"eval xong {k}year")
    pd.DataFrame(cost_rows).to_csv(os.path.join(args.out, "cost_results.csv"), index=False)
    pd.DataFrame(bench_rows).to_csv(os.path.join(args.out, "cost_benchmarks.csv"), index=False)
    pd.DataFrame(ci_rows).to_csv(os.path.join(args.out, "bootstrap_ci.csv"), index=False)
    dr = pd.DataFrame(diff_rows)
    dr.to_csv(os.path.join(args.out, "paired_differences.csv"), index=False)
    # [7] Đối chiếu ECE10 cũ (legacy) và mới cho mọi so sánh hiệu chỉnh - thô
    ec = dr[(dr.question == "RQ2: calibrated - raw") & dr.metric.isin(["ece10", "ece10_legacy"])]
    ec.to_csv(os.path.join(args.out, "ece_sign_check.csv"), index=False)
    log(f"ECE sign check: legacy mâu thuẫn dấu {int(ec[ec.metric=='ece10_legacy'].sign_mismatch.sum())} ca; "
        f"bản sửa {int(ec[ec.metric=='ece10'].sign_mismatch.sum())} ca")
    if args.orig_out and os.path.exists(os.path.join(args.orig_out, "test_predictions_1year.csv")):
        ece_check_original(args)


def ece_check_original(args):
    """[7] Tái lập chênh lệch ECE10 (hiệu chỉnh - thô) trên dự báo test của bản GỐC (chỉ đọc out/),
    bằng ECE cũ và ECE mới với cùng chỉ số bootstrap như bản gốc."""
    rows = []
    for k in args.files:
        pth = os.path.join(args.orig_out, f"test_predictions_{k}year.csv")
        if not os.path.exists(pth): continue
        d = pd.read_csv(pth); y = d["y_true"].values.astype(int)
        idx = boot_indices(y, k, args.n_boot)
        for m in ("lr", "xgb"):
            for wn in WEIGHT_MODES:
                raw = d[f"{m}_{wn}_none"].values
                for c in ("platt", "isotonic"):
                    p = d[f"{m}_{wn}_{c}"].values
                    for name, f in (("legacy", ece_quantile_legacy), ("fixed", lambda yy, pp, nb: _ece_fixed(yy, pp, nb))):
                        bd = np.array([f(y[i], p[i], 10) - f(y[i], raw[i], 10) for i in idx])
                        lo, hi = np.percentile(bd, [2.5, 97.5]); pdiff = f(y, p, 10) - f(y, raw, 10)
                        rows.append(dict(file=f"{k}year", model=m, weighting=wn, calibration=c, ece_method=name,
                                         point_diff=pdiff, ci_low=lo, ci_high=hi,
                                         sign_mismatch=bool((pdiff < 0 and lo > 0) or (pdiff > 0 and hi < 0))))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "ece_sign_check_original_out.csv"), index=False)
    log("đã đối chiếu ECE trên dự báo của bản gốc: ece_sign_check_original_out.csv")


def _ece_fixed(y, p, nb):
    global LEGACY_ECE
    old = LEGACY_ECE; LEGACY_ECE = False
    try:
        return ece_quantile(y, p, nb)
    finally:
        LEGACY_ECE = old


def stage_shap(args):
    import shap
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pdir = os.path.join(args.out, "figures"); os.makedirs(pdir, exist_ok=True)
    rows, fam_rows, a27 = [], [], []
    xgb_models = [m for m in args.models if m in XGB_MODELS]
    for model in xgb_models:
        for mode in WEIGHT_MODES:
            w = mode == "weighted"
            for k in args.files:
                mp = os.path.join(args.out, "models", f"{model}_{k}year_{'w' if w else 'u'}.joblib")
                if not os.path.exists(mp): continue
                pipe = joblib.load(mp)
                Xtr, Xte, ytr, yte = load_split(args.data_dir, k, args.dedup)
                variants = [(model, pipe)]
                # Ablation cô lập tác động của chỉ báo thiếu: cùng siêu tham số, cùng train, add_indicator=False
                if model == "xgb" and "imputer" in pipe.named_steps and pipe.named_steps["imputer"].add_indicator:
                    abl = clone(pipe).set_params(imputer__add_indicator=False).fit(Xtr, ytr)
                    variants.append(("xgb_noind_refit", abl))
                for vname, vp in variants:
                    Xt = transform_for_trees(vp, Xte)
                    booster = vp.named_steps["clf"].model_
                    expl = shap.TreeExplainer(booster)
                    sv = np.asarray(expl.shap_values(Xt.values))       # log-odds của mô hình cơ sở CHƯA hiệu chỉnh
                    if sv.ndim == 3: sv = sv[..., 1]
                    mean_abs = np.abs(sv).mean(axis=0); cols = list(Xt.columns); tot = mean_abs.sum()
                    for f, v in zip(cols, mean_abs):
                        rows.append(dict(model=vname, mode=mode, file=f"{k}year", feature=f, family=family_of(f),
                                         mean_abs_shap=v, share=v / tot))
                    fams = pd.Series(mean_abs, index=cols).groupby([family_of(c) for c in cols]).agg(["sum", "size"])
                    for fam, rr in fams.iterrows():
                        fam_rows.append(dict(model=vname, mode=mode, file=f"{k}year", family=fam, n_features=int(rr["size"]),
                                             total_family_share=rr["sum"] / tot,
                                             avg_contribution_per_feature=rr["sum"] / tot / rr["size"]))
                    s = pd.Series(mean_abs, index=cols); rk = s.rank(ascending=False, method="min")
                    p_te = vp.predict_proba(Xte)[:, 1]
                    a27.append(dict(model=vname, mode=mode, file=f"{k}year", test_auc_raw=roc_auc_score(yte, p_te),
                                    attr27_mean_abs=s.get("Attr27", np.nan), attr27_rank=rk.get("Attr27", np.nan),
                                    attr27_share=s.get("Attr27", np.nan) / tot,
                                    miss_attr27_mean_abs=s.get("miss_Attr27", 0.0),
                                    miss_attr27_rank=rk.get("miss_Attr27", np.nan),
                                    attr27_plus_miss_share=(s.get("Attr27", 0) + s.get("miss_Attr27", 0)) / tot,
                                    n_columns=len(cols), top1=s.idxmax()))
                    if vname == SHAP_PRIMARY_MODEL and mode == SHAP_PRIMARY_MODE:
                        plt.figure(); shap.summary_plot(sv, Xt, max_display=15, show=False)
                        plt.title(f"{k}year - SHAP beeswarm (log-odds, XGB {mode})"); plt.tight_layout()
                        plt.savefig(os.path.join(pdir, f"phuluc_shap_beeswarm_{k}year.png"), dpi=150); plt.close()
                        # Waterfall minh họa: TP = bankrupt có p cao nhất (>= t*), FN = bankrupt có p thấp nhất (< t*)
                        base = float(np.ravel(expl.expected_value)[0])
                        cand = {"TP": np.where((yte == 1) & (p_te >= T_STAR))[0], "FN": np.where((yte == 1) & (p_te < T_STAR))[0]}
                        for name, ix in cand.items():
                            if len(ix) == 0: continue
                            i = ix[np.argmax(p_te[ix])] if name == "TP" else ix[np.argmin(p_te[ix])]
                            plt.figure()
                            shap.plots.waterfall(shap.Explanation(values=sv[i], base_values=base, data=Xt.iloc[i].values,
                                                                  feature_names=cols), max_display=12, show=False)
                            plt.title(f"{k}year - waterfall {name} (p={p_te[i]:.3f})"); plt.tight_layout()
                            plt.savefig(os.path.join(pdir, f"shap_waterfall_{name}_{k}year.png"), dpi=150, bbox_inches="tight"); plt.close()
                log(f"SHAP xong {model} {mode} {k}year")
    if not rows:
        log("Không có mô hình XGBoost đã fit (cần chạy stage fit với xgboost)."); return
    sh = pd.DataFrame(rows); fam = pd.DataFrame(fam_rows)
    sh.to_csv(os.path.join(args.out, "shap_mean_abs.csv"), index=False)
    fam.to_csv(os.path.join(args.out, "shap_family.csv"), index=False)
    pd.DataFrame(a27).to_csv(os.path.join(args.out, "shap_attr27.csv"), index=False)
    sh["rank"] = sh.groupby(["model", "mode", "file"])["mean_abs_shap"].rank(ascending=False, method="min")
    sh[sh["rank"] <= 10].sort_values(["model", "mode", "file", "rank"]).to_csv(os.path.join(args.out, "shap_top10.csv"), index=False)
    # Nhất quán giữa các file: Spearman trên 64 đặc trưng gốc + trùng lặp top-10 (trên mọi cột, kể cả chỉ báo thiếu)
    cons, leak = [], []
    for (model, mode), g in sh.groupby(["model", "mode"]):
        sa = g.pivot(index="feature", columns="file", values="mean_abs_shap").fillna(0.0)
        s = sa.loc[FEATURES]
        gf = fam[(fam.model == model) & (fam["mode"] == mode)]
        f = gf.pivot(index="family", columns="file", values="total_family_share").fillna(0.0)
        fa = gf.pivot(index="family", columns="file", values="avg_contribution_per_feature").fillna(0.0)
        files = list(s.columns)
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                a, b = files[i], files[j]
                top = lambda col: set(sa[col].nlargest(10).index)
                cons.append(dict(model=model, mode=mode, file_A=a, file_B=b,
                                 spearman_feature=spearmanr(s[a], s[b])[0],
                                 top10_overlap=len(top(a) & top(b)) / 10,
                                 spearman_family_total_share=spearmanr(f[a], f[b])[0],
                                 spearman_family_avg_per_feature=spearmanr(fa[a], fa[b])[0]))
        tops = {c: sa[c].idxmax() for c in files}
        shares = {c: sa[c].max() / sa[c].sum() for c in files}
        for c in files:
            leak.append(dict(model=model, mode=mode, file=c, top1_feature=tops[c], top1_share_of_total=shares[c],
                             same_top1_in_all_files=len(set(tops.values())) == 1,
                             flag_possible_leakage=(len(set(tops.values())) == 1 and min(shares.values()) > 0.25)))
    pd.DataFrame(cons).to_csv(os.path.join(args.out, "shap_file_pair_consistency.csv"), index=False)
    pd.DataFrame(leak).to_csv(os.path.join(args.out, "leakage_check.csv"), index=False)
    log("SHAP xong")


# ----------------------------------------------------------------------------------------------
# 6. BẢNG & HÌNH CHO BÁO CÁO (out/tables, out/figures)
# ----------------------------------------------------------------------------------------------
# Bảng màu phân loại (thứ tự cố định, không lặp vòng)
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CAL_COLOR = {"none": PAL[0], "platt": PAL[1], "isotonic": PAL[2]}
CAL_VI = {"none": "Thô", "platt": "Platt", "isotonic": "Isotonic"}
TS_VI = {"unweighted": "Không", "weighted": "Có"}


def model_label(m, args):
    return {"lr": "LR", "lr_robust": "LR+QT" if args.lr_robust == "quantile" else "LR+Win",
            "xgb": "XGB", "xgb_nan": "XGB-NaN"}[m]


def vn(x, d=3):
    """Định dạng số kiểu Việt Nam: dấu phẩy thập phân."""
    if x is None or (isinstance(x, float) and np.isnan(x)): return ""
    return f"{x:.{d}f}".replace(".", ",")


def vn_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


def _setup_mpl():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                         "axes.labelcolor": "#2b2b29", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "font.size": 9, "legend.frameon": False, "axes.axisbelow": True})
    return plt


def stage_report(args):
    tdir = os.path.join(args.out, "tables"); fdir = os.path.join(args.out, "figures")
    os.makedirs(tdir, exist_ok=True); os.makedirs(fdir, exist_ok=True)
    def rd(n):
        pth = os.path.join(args.out, n)
        return pd.read_csv(pth) if os.path.exists(pth) and os.path.getsize(pth) > 1 else None
    eda, mp, bp, ci, dr = rd("eda_summary.csv"), rd("metrics_point.csv"), rd("best_params.csv"), rd("bootstrap_ci.csv"), rd("paired_differences.csv")
    cr, cb, sh, fam, cons, rep = (rd("cost_results.csv"), rd("cost_benchmarks.csv"), rd("shap_mean_abs.csv"),
                                  rd("shap_family.csv"), rd("shap_file_pair_consistency.csv"), rd("repeated_split_summary.csv"))
    files = [f"{k}year" for k in args.files]
    configs = [(m, w) for m in args.models for w in WEIGHT_MODES]
    cfg_col = lambda m, w: f"{model_label(m, args)} {'có TS' if w == 'weighted' else 'không TS'}"
    save = lambda df, name: df.to_csv(os.path.join(tdir, name), index=False, encoding="utf-8-sig")
    plt = _setup_mpl()

    # Bảng 1: EDA
    if eda is not None:
        b1 = pd.DataFrame({"Tệp": eda.file, "Thời hạn (năm)": eda.horizon_years_assumed, "N gốc": eda.n_raw.map(vn_int),
                           "Dòng trùng (loại)": eda.dup_rows_with_label, "N sau loại trùng": eda.n.map(vn_int),
                           "Phá sản": eda.n_pos, "Tỷ lệ (%)": eda.prevalence_pct.map(lambda v: vn(v, 2)),
                           "Ô thiếu (%)": eda.missing_cells_pct.map(lambda v: vn(v, 2)),
                           "Dòng ≥1 thiếu (%)": eda.rows_any_missing_pct.map(lambda v: vn(v, 1)),
                           "Train / Test": eda.n_train.astype(str) + "/" + eda.n_test.astype(str),
                           "Dương (test)": eda.pos_test, "Test trùng train": eda.test_rows_duplicated_in_train,
                           "Attr27 thiếu (%) PS / không PS": eda.attr27_missing_pct_pos.map(lambda v: vn(v, 1)) + " / "
                                                             + eda.attr27_missing_pct_neg.map(lambda v: vn(v, 1))})
        save(b1, "bang01_eda.csv")

    # Bảng 2: siêu tham số
    if bp is not None:
        def fmt_bp(r):
            if r.model.startswith("lr"):
                return f"C={float(r['clf__C']):.4g}; LL={vn(r.cv_logloss)}".replace(".", ",")
            g = lambda c: r.get(c)
            return (f"d={g('clf__max_depth')}, η={g('clf__learning_rate')}, n={g('clf__n_estimators')}, "
                    f"λ={g('clf__reg_lambda')}, cs={g('clf__colsample_bytree')}, ss={g('clf__subsample')}, "
                    f"mcw={g('clf__min_child_weight')}; LL={vn(r.cv_logloss)}").replace(".", ",")
        bp["cell"] = bp.apply(fmt_bp, axis=1)
        b2 = bp.pivot(index="file", columns=["model", "weighting"], values="cell")
        b2 = b2[[c for c in configs if c in b2.columns]]
        b2.columns = [cfg_col(m, w) for m, w in b2.columns]
        save(b2.reset_index().rename(columns={"file": "Tệp"}), "bang02_sieu_tham_so.csv")

    # Bảng 3, 4: AUC, AP (thô) điểm [KTC 95%]
    if ci is not None and eda is not None:
        prev = dict(zip(eda.file, eda.prevalence_pct / 100))
        for num, met, nm in ((3, "auc", "auc"), (4, "ap", "ap")):
            c = ci[(ci.metric == met) & (ci.calibration == "none")]
            rows = []
            for f in files:
                row = {"Tệp": f"{f} ({HORIZON_YEARS[int(f[0])]} năm)", "Tỷ lệ": vn(prev[f], 4)}
                for m, w in configs:
                    r = c[(c.file == f) & (c.model == m) & (c.weighting == w)]
                    if len(r): r = r.iloc[0]; row[cfg_col(m, w)] = f"{vn(r.point)} [{vn(r.ci_low, 2)}; {vn(r.ci_high, 2)}]"
                rows.append(row)
            save(pd.DataFrame(rows), f"bang0{num}_{nm}_test_tho.csv")

    # Bảng 5, 6 (+ lưới phụ): chỉ số theo toàn lưới mô hình × TS × hiệu chỉnh
    if mp is not None:
        for name, met, d in (("bang05_brier", "brier", 4), ("bang06_ece10", "ece10", 4), ("grid_ece5", "ece5", 4),
                             ("grid_bss", "bss", 4), ("grid_logloss", "logloss", 4),
                             ("grid_cal_slope", "cal_slope", 3), ("grid_cal_intercept", "cal_intercept", 3)):
            rows = []
            for m, w in configs:
                for c in CALS:
                    r = {"Mô hình": model_label(m, args), "TS": TS_VI[w], "Xác suất": CAL_VI[c]}
                    for f in files:
                        v = mp[(mp.file == f) & (mp.model == m) & (mp.weighting == w) & (mp.calibration == c)][met]
                        r[f] = vn(v.iloc[0], d) if len(v) else ""
                    rows.append(r)
            save(pd.DataFrame(rows), f"{name}.csv")

    # Bảng 7, 10: đếm tốt hơn / không khác biệt / xấu hơn (theo KTC 95%)
    if dr is not None:
        def count_table(question, metrics, name, first_col_name):
            rows = []
            for m, w in configs:
                for c in ("platt", "isotonic"):
                    r = {"Mô hình": model_label(m, args), "TS": TS_VI[w], "Hiệu chỉnh": CAL_VI[c]}
                    for met in metrics:
                        q = dr[(dr.question == question) & (dr.metric == met) & (dr.A == f"{m}|{w}|{c}")]
                        r[first_col_name.get(met, met)] = (f"{(q.verdict == 'A_better').sum()} / {(q.verdict == 'no_diff').sum()} / "
                                                           f"{(q.verdict == 'A_worse').sum()}")
                    rows.append(r)
            save(pd.DataFrame(rows), name)
        count_table("RQ2: calibrated - raw", ["brier", "ece10", "ece5", "bss", "logloss"], "bang07_dem_hieu_chinh.csv",
                    {"brier": "Brier", "ece10": "ECE10", "ece5": "ECE5", "bss": "BSS", "logloss": "Log loss"})
        count_table("RQ4: cost calibrated - raw (P1, ratio 10:1)", ["cost_p1"], "bang10_dem_chi_phi_P1.csv",
                    {"cost_p1": "P1 (chính): thấp hơn / không khác / cao hơn"})

    # Bảng 8: Recall, Precision, FN, FP tại t* (thô)
    if mp is not None:
        q = mp[mp.calibration == "none"]
        rows = []
        for f in files:
            for m, w in configs:
                r = q[(q.file == f) & (q.model == m) & (q.weighting == w)]
                if not len(r): continue
                r = r.iloc[0]
                rows.append({"Tệp": f, "Mô hình": model_label(m, args), "TS": TS_VI[w], "Recall": vn(r.recall),
                             "Precision": vn(r.precision), "FN": int(r.fn), "FP": int(r.fp)})
        save(pd.DataFrame(rows), "bang08_recall_precision_tstar.csv")

    # Bảng 9: tổng chi phí P1, P2 (10:1)
    if cr is not None:
        q = cr[cr.ratio == "10:1"]
        rows = []
        for m, w in configs:
            for c in CALS:
                r = {"Mô hình": model_label(m, args), "TS": TS_VI[w], "Xác suất": CAL_VI[c]}
                for pol in ("P1", "P2"):
                    for f in files:
                        v = q[(q.file == f) & (q.model == m) & (q.weighting == w) & (q.calibration == c) & (q.policy == pol)]
                        r[f"{pol} {f[0]}y"] = vn_int(v.total_cost.iloc[0]) if len(v) else ""
                rows.append(r)
        bm = cb[cb.ratio == "10:1"]
        for name, lab in (("approve_all", "Không gắn cờ ai (mốc)"), ("reject_all", "Gắn cờ tất cả (mốc)")):
            r = {"Mô hình": lab, "TS": "", "Xác suất": ""}
            for f in files:
                v = bm[(bm.file == f) & (bm.benchmark == name)].total_cost
                r[f"P1 {f[0]}y"] = vn_int(v.iloc[0]) if len(v) else ""
            rows.append(r)
        save(pd.DataFrame(rows), "bang09_chi_phi_10_1.csv")

    # Bảng 11: độ nhạy theo tỷ lệ chi phí (so sánh điểm, P1)
    if cr is not None:
        q = cr[cr.policy == "P1"]
        rows = []
        for ratio in COST_RATIOS:
            rs = f"{ratio}:1"; qq = q[q.ratio == rs]
            raw = qq[qq.calibration == "none"].set_index(["file", "model", "weighting"]).total_cost
            cal = qq[qq.calibration != "none"].set_index(["file", "model", "weighting", "calibration"]).total_cost
            d = (cal - raw.reindex(cal.index.droplevel("calibration")).values).reset_index()
            cnt = lambda x: f"{(x.total_cost < 0).sum()} / {(x.total_cost == 0).sum()} / {(x.total_cost > 0).sum()}"
            excl = d[~((d.weighting == "weighted") & d.model.str.startswith("lr"))]
            rows.append({"C_FN : C_FP": rs, f"Tất cả ({len(d)} cặp)": cnt(d),
                         f"Loại LR có TS ({len(excl)} cặp)": cnt(excl)})
        save(pd.DataFrame(rows), "bang11_do_nhay_ty_le_chi_phi.csv")

    # Bảng 12-14: SHAP (mô hình chính, không trọng số)
    if sh is not None:
        abbr = lambda fam_: fam_[:4] + "."
        s = sh[(sh.model == SHAP_PRIMARY_MODEL) & (sh["mode"] == SHAP_PRIMARY_MODE)].copy()
        s["rank"] = s.groupby("file").mean_abs_shap.rank(ascending=False, method="first")
        b12 = {"Hạng": list(range(1, 11))}
        for f in files:
            t = s[s.file == f].sort_values("rank").head(10)
            b12[f] = [f"{r.feature} ({abbr(r.family)}) {vn(r.mean_abs_shap, 2)}" for r in t.itertuples()]
        save(pd.DataFrame(b12), "bang12_top10_shap.csv")
        if fam is not None:
            ff = fam[(fam.model == SHAP_PRIMARY_MODEL) & (fam["mode"] == SHAP_PRIMARY_MODE)]
            rows = []
            for fa_ in list(FAMILY_MAP) + [MISSING_FAMILY]:
                g = ff[ff.family == fa_]
                if not len(g): continue
                r = {"Họ": fa_, "Số ĐT": "/".join(sorted({str(v) for v in g.n_features}))}
                for f in files:
                    v = g[g.file == f]
                    r[f] = f"{vn(v.total_family_share.iloc[0])} / {vn(v.avg_contribution_per_feature.iloc[0], 4)}" if len(v) else ""
                rows.append(r)
            save(pd.DataFrame(rows), "bang13_ho_dac_trung.csv")
        if cons is not None:
            cc = cons[cons.model == SHAP_PRIMARY_MODEL]
            u = cc[cc["mode"] == "unweighted"].set_index(["file_A", "file_B"]); wv = cc[cc["mode"] == "weighted"].set_index(["file_A", "file_B"])
            rows = []
            for (a, b) in u.index:
                r = {"Cặp tệp": f"{a} – {b}"}
                for lab, src in (("", u), (" (w)", wv)):
                    if (a, b) not in src.index: continue
                    x = src.loc[(a, b)]
                    r.update({f"ρ_ĐT{lab}": vn(x.spearman_feature, 2), f"T10{lab}": vn(x.top10_overlap, 1),
                              f"ρ_TT{lab}": vn(x.spearman_family_total_share, 2), f"ρ_TB{lab}": vn(x.spearman_family_avg_per_feature, 2)})
                rows.append(r)
            save(pd.DataFrame(rows), "bang14_nhat_quan_shap.csv")

    # Bảng 15 (mới): lặp 5 phép chia 80/20 - trung bình ± độ lệch chuẩn (xác suất thô)
    if rep is not None:
        pm = lambda r, c, d=3: f"{vn(r[c + '_mean'], d)} ± {vn(r[c + '_sd'], d)}"
        rows = []
        for f in files:
            for m, w in configs:
                for c in CALS:
                    r = rep[(rep.file == f) & (rep.model == m) & (rep.weighting == w) & (rep.calibration == c)
                            & (rep.n_splits >= 2)]                 # cần >= 2 phép chia để có độ lệch chuẩn
                    if not len(r): continue
                    r = r.iloc[0]
                    rows.append({"Tệp": f, "Mô hình": model_label(m, args), "TS": TS_VI[w], "Xác suất": CAL_VI[c],
                                 "Số phép chia": int(r.n_splits), "AUC": pm(r, "auc"), "AP": pm(r, "ap"),
                                 "Brier": pm(r, "brier", 4), "BSS": pm(r, "bss"), "Log loss": pm(r, "logloss"),
                                 "ECE10": pm(r, "ece10", 4), "Slope": pm(r, "cal_slope", 2),
                                 "Intercept": pm(r, "cal_intercept", 2), "Chi phí P1": pm(r, "cost_p1", 0)})
        b15 = pd.DataFrame(rows)
        save(b15[b15["Xác suất"] == "Thô"], "bang15_lap_lai_5_phep_chia.csv")
        save(b15, "bang15b_lap_lai_5_phep_chia_day_du.csv")
    gs = rd("repeated_split_gap_summary.csv")
    if gs is not None:
        gs = gs[gs.auc_diff_count >= 2].copy()
        gs["Cặp"] = gs.A.map(lambda m: model_label(m, args)) + " − " + gs.B.map(lambda m: model_label(m, args))
        gs["ΔAUC (TB ± SD)"] = gs.apply(lambda r: f"{vn(r.auc_diff_mean)} ± {vn(r.auc_diff_std)}", axis=1)
        gs["ΔAP (TB ± SD)"] = gs.apply(lambda r: f"{vn(r.ap_diff_mean)} ± {vn(r.ap_diff_std)}", axis=1)
        gs["TS"] = gs.weighting.map(TS_VI)
        save(gs[["file", "TS", "Cặp", "ΔAUC (TB ± SD)", "ΔAP (TB ± SD)", "auc_diff_count"]]
             .rename(columns={"file": "Tệp", "auc_diff_count": "Số phép chia"}), "bang16_khoang_cach_auc.csv")
    log("bảng xong: " + ", ".join(sorted(os.listdir(tdir))))

    # ---------------- HÌNH ----------------
    xs = np.arange(len(files))
    # Hình 1: AUC & AP (thô) với KTC 95%; hàng = trọng số, cột = chỉ số
    if ci is not None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
        for i, w in enumerate(WEIGHT_MODES):
            for j, met in enumerate(("auc", "ap")):
                ax = axes[i, j]; c = ci[(ci.metric == met) & (ci.calibration == "none") & (ci.weighting == w)]
                for mi, m in enumerate(args.models):
                    r = c[c.model == m].set_index("file").reindex(files)
                    off = (mi - (len(args.models) - 1) / 2) * 0.17
                    ax.errorbar(xs + off, r.point, yerr=[r.point - r.ci_low, r.ci_high - r.point], fmt="o", ms=5,
                                color=PAL[mi], ecolor=PAL[mi], elinewidth=1.5, capsize=0, label=model_label(m, args))
                ax.set_title(f"{met.upper()} - {'có trọng số' if w == 'weighted' else 'không trọng số'}", loc="left", fontsize=10)
                ax.set_xticks(xs, files); ax.set_ylim(0, 1.02)
        axes[0, 0].legend(ncol=len(args.models), loc="lower left", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(fdir, "hinh1_auc_ap_ci.png"), dpi=150); plt.close(fig)

    # Hình 2: biểu đồ độ tin cậy (quantile, 10 bin) - một hình cho mỗi chế độ trọng số
    for w in WEIGHT_MODES:
        fig, axes = plt.subplots(len(files), len(args.models), figsize=(3.3 * len(args.models), 3.0 * len(files)), squeeze=False)
        for i, f in enumerate(files):
            pth = os.path.join(args.out, f"test_predictions_{f}.csv")
            if not os.path.exists(pth): continue
            d = pd.read_csv(pth)
            for j, m in enumerate(args.models):
                ax = axes[i, j]; col0 = f"{m}_{w}_none"
                if col0 not in d: continue
                hi = 0
                for c in CALS:
                    fp_, mp_ = calibration_curve(d["y_true"], d[f"{m}_{w}_{c}"], n_bins=10, strategy="quantile")
                    ax.plot(mp_, fp_, marker="o", ms=3.5, lw=1.8, color=CAL_COLOR[c], label=CAL_VI[c])
                    hi = max(hi, mp_.max(), fp_.max())
                ax.plot([0, hi], [0, hi], color="#8a8984", ls="--", lw=0.8)
                ax.set_title(f"{f} | {model_label(m, args)}", loc="left", fontsize=9)
                if i == len(files) - 1: ax.set_xlabel("p dự báo (TB bin)")
                if j == 0: ax.set_ylabel("Tần suất thực")
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(fdir, f"hinh2_reliability_{w}.png"), dpi=130); plt.close(fig)

    # Hình 3: tổng chi phí P1 (10:1) theo cấu hình & hiệu chỉnh, so với hai mốc
    if cr is not None:
        q = cr[(cr.ratio == "10:1") & (cr.policy == "P1")]; bm = cb[cb.ratio == "10:1"]
        fig, axes = plt.subplots(len(files), 1, figsize=(10, 2.6 * len(files)), squeeze=False)
        xc = np.arange(len(configs)); bw = 0.26
        for i, f in enumerate(files):
            ax = axes[i, 0]
            for ci_, c in enumerate(CALS):
                v = [q[(q.file == f) & (q.model == m) & (q.weighting == w) & (q.calibration == c)].total_cost.sum() for m, w in configs]
                ax.bar(xc + (ci_ - 1) * bw, v, width=bw - 0.03, color=CAL_COLOR[c], label=CAL_VI[c])
            for name, ls, lab in (("approve_all", "--", "Không gắn cờ ai"), ("reject_all", ":", "Gắn cờ tất cả")):
                v = bm[(bm.file == f) & (bm.benchmark == name)].total_cost
                if len(v): ax.axhline(v.iloc[0], color="#52514e", ls=ls, lw=1, label=lab)
            ax.set_title(f, loc="left", fontsize=10); ax.set_ylabel("Tổng chi phí")
            ax.set_xticks(xc, [cfg_col(m, w) for m, w in configs], fontsize=8)
        axes[0, 0].legend(ncol=5, fontsize=8, loc="upper right")
        fig.tight_layout(); fig.savefig(os.path.join(fdir, "hinh3_chi_phi_P1.png"), dpi=150); plt.close(fig)

    # [8] Hình 8 (mới): cost curve - tổng chi phí theo ngưỡng 0,01-0,50, đánh dấu t* = 1/11
    if args.cost_curve:
        grid = np.round(np.arange(0.01, 0.5001, 0.005), 4); rows = []
        for w in WEIGHT_MODES:
            fig, axes = plt.subplots(len(files), len(args.models), figsize=(3.3 * len(args.models), 2.8 * len(files)), squeeze=False)
            for i, f in enumerate(files):
                pth = os.path.join(args.out, f"test_predictions_{f}.csv")
                if not os.path.exists(pth): continue
                d = pd.read_csv(pth); y = d.y_true.values
                for j, m in enumerate(args.models):
                    ax = axes[i, j]
                    for c in CALS:
                        col = f"{m}_{w}_{c}"
                        if col not in d: continue
                        p = d[col].values; cost = np.array([cost_at(y, p, t, PRIMARY_RATIO) for t in grid])
                        rows += [dict(file=f, model=m, weighting=w, calibration=c, threshold=t, total_cost=v) for t, v in zip(grid, cost)]
                        ax.plot(grid, cost, color=CAL_COLOR[c], lw=1.8, label=CAL_VI[c])
                        ax.plot([T_STAR], [cost_at(y, p, T_STAR, PRIMARY_RATIO)], "o", ms=6, color=CAL_COLOR[c],
                                markeredgecolor="white", markeredgewidth=1.5)
                    ax.axvline(T_STAR, color="#52514e", ls="--", lw=0.9)
                    ax.axhline(PRIMARY_RATIO * y.sum(), color="#8a8984", ls=":", lw=0.9)
                    ax.set_title(f"{f} | {model_label(m, args)}", loc="left", fontsize=9)
                    if i == len(files) - 1: ax.set_xlabel("Ngưỡng t")
                    if j == 0: ax.set_ylabel("Tổng chi phí (10:1)")
            axes[0, 0].legend(fontsize=7, title="t* = 1/11 (nét đứt)", title_fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(fdir, f"hinh8_cost_curve_{w}.png"), dpi=130); plt.close(fig)
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "cost_curve.csv"), index=False)

    # Hình 4-6: SHAP (mô hình chính, không trọng số)
    if sh is not None:
        import seaborn as sns
        s = sh[(sh.model == SHAP_PRIMARY_MODEL) & (sh["mode"] == SHAP_PRIMARY_MODE)]
        pv = s.pivot(index="feature", columns="file", values="mean_abs_shap").fillna(0.0)[files]
        rk = pv.rank(ascending=False, method="min")
        union = sorted({f for c in files for f in pv[c].nlargest(10).index}, key=lambda f: rk.loc[f].mean())
        fig, ax = plt.subplots(figsize=(6, 0.32 * len(union) + 1.5))
        sns.heatmap(rk.loc[union], cmap="Blues_r", annot=True, fmt=".0f", annot_kws={"size": 7}, linewidths=1,
                    linecolor="white", cbar_kws={"label": "Hạng (1 = quan trọng nhất)"}, ax=ax)
        ax.set_title("Hạng mean|SHAP| (64 đặc trưng + chỉ báo thiếu), hợp top-10", loc="left", fontsize=10)
        ax.set_xlabel(""); ax.set_ylabel(""); fig.tight_layout()
        fig.savefig(os.path.join(fdir, "hinh4_shap_rank_heatmap.png"), dpi=150); plt.close(fig)
        if fam is not None:
            ff = fam[(fam.model == SHAP_PRIMARY_MODEL) & (fam["mode"] == SHAP_PRIMARY_MODE)]
            order = [x for x in list(FAMILY_MAP) + [MISSING_FAMILY] if x in set(ff.family)]
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
            for ax, col, ttl, fmt in ((axes[0], "total_family_share", "Tổng tỷ trọng họ", ".2f"),
                                      (axes[1], "avg_contribution_per_feature", "Đóng góp TB / đặc trưng", ".3f")):
                sns.heatmap(ff.pivot(index="family", columns="file", values=col).loc[order, files], cmap="Blues",
                            annot=True, fmt=fmt, annot_kws={"size": 8}, linewidths=1, linecolor="white", ax=ax)
                ax.set_title(ttl, loc="left", fontsize=10); ax.set_xlabel(""); ax.set_ylabel("")
            fig.tight_layout(); fig.savefig(os.path.join(fdir, "hinh5_ho_dac_trung.png"), dpi=150); plt.close(fig)
        sp = pd.DataFrame(np.eye(len(files)), index=files, columns=files)
        for a in files:
            for b in files:
                if a != b: sp.loc[a, b] = spearmanr(pv.loc[FEATURES, a], pv.loc[FEATURES, b])[0]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(sp, cmap="Blues", vmin=0, vmax=1, annot=True, fmt=".2f", linewidths=1, linecolor="white", ax=ax,
                    cbar_kws={"label": "Spearman ρ"})
        ax.set_title("Spearman hạng mean|SHAP| (64 đặc trưng)", loc="left", fontsize=10); fig.tight_layout()
        fig.savefig(os.path.join(fdir, "hinh6_spearman_tep.png"), dpi=150); plt.close(fig)
        # Hình 7: ghép waterfall TP/FN của 1year, 2year
        wf = [[os.path.join(fdir, f"shap_waterfall_{n}_{f}.png") for n in ("TP", "FN")] for f in files[:2]]
        if all(os.path.exists(p) for r in wf for p in r):
            import matplotlib.image as mpimg
            fig, axes = plt.subplots(len(wf), 2, figsize=(14, 5.2 * len(wf)), squeeze=False)
            for i, r in enumerate(wf):
                for j, p in enumerate(r):
                    axes[i, j].imshow(mpimg.imread(p)); axes[i, j].axis("off")
            fig.tight_layout(); fig.savefig(os.path.join(fdir, "hinh7_waterfall.png"), dpi=130); plt.close(fig)
    log("hình xong: " + ", ".join(sorted(os.listdir(fdir))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(BASE_DIR, "DatasetGK"))   # thư mục chứa 1year.arff ... 5year.arff
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "out_v2"))
    ap.add_argument("--orig-out", default=os.path.join(BASE_DIR, "out"), help="thư mục kết quả gốc (CHỈ ĐỌC) để đối chiếu ECE")
    ap.add_argument("--stage", default="all", choices=["plan", "eda", "fit", "eval", "shap", "report", "all"])
    ap.add_argument("--models", default=None, help="mặc định: lr,lr_robust,xgb,xgb_nan (theo các cờ bên dưới)")
    ap.add_argument("--files", default="1,2,3,4,5", help="vd: 1 để chạy thử trên 1year.arff")
    ap.add_argument("--n-jobs", type=int, default=-1); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    # các thay đổi v2 (bật/tắt được)
    ap.add_argument("--no-add-indicator", dest="add_indicator", action="store_false", help="[1] tắt add_indicator")
    ap.add_argument("--no-xgb-nan", dest="xgb_nan", action="store_false", help="[1] bỏ biến thể XGB để NaN")
    ap.add_argument("--lr-robust", default="quantile", choices=["quantile", "winsor", "none"], help="[2] biến thể LR mạnh hơn")
    ap.add_argument("--no-dedup", dest="dedup", action="store_false", help="[3] không loại dòng trùng")
    ap.add_argument("--search", default="wide", choices=["wide", "orig"], help="[4] lưới tìm kiếm siêu tham số")
    ap.add_argument("--n-cand", type=int, default=N_CAND, help="[4] số tổ hợp cố định khi --search wide")
    ap.add_argument("--repeats", type=int, default=N_REPEATS, help="[5] số phép chia 80/20 (1 = chỉ phép chia chính)")
    ap.add_argument("--repeat-models", default=None,
                    help="[5] mô hình chạy ở các phép chia lặp lại (seed != 42); mặc định: như --models")
    ap.add_argument("--ece-legacy", action="store_true", help="[7] dùng ECE cũ (cắt ngang ties)")
    ap.add_argument("--no-cost-curve", dest="cost_curve", action="store_false", help="[8] không vẽ cost curve")
    ap.add_argument("--n-boot", type=int, default=None)
    args = ap.parse_args()
    if args.models is None:
        ms = ["lr"] + (["lr_robust"] if args.lr_robust != "none" else []) + ["xgb"] + (["xgb_nan"] if args.xgb_nan else [])
        args.models = tuple(ms)
    else:
        args.models = tuple(args.models.split(","))
    args.repeat_models = args.models if args.repeat_models is None else tuple(args.repeat_models.split(","))
    args.files = [int(x) for x in args.files.split(",")]
    if args.n_boot is None: args.n_boot = 30 if args.smoke else N_BOOT
    if args.smoke: args.repeats = min(args.repeats, 2)
    global LEGACY_ECE
    LEGACY_ECE = args.ece_legacy
    if any(m in XGB_MODELS for m in args.models) and XGBClassifier is None:
        sys.exit("LỖI: thiếu 'xgboost'. Cài đặt (pip install -r requirements.txt) rồi chạy lại.")
    # Tự dò thư mục dữ liệu: --data-dir -> cùng thư mục với script -> BASE_DIR/DatasetGK
    for cand in (args.data_dir, BASE_DIR, os.path.join(BASE_DIR, "DatasetGK")):
        if all(os.path.exists(os.path.join(cand, f"{k}year.arff")) for k in args.files):
            args.data_dir = cand; break
    missing = [f"{k}year.arff" for k in args.files if not os.path.exists(os.path.join(args.data_dir, f"{k}year.arff"))]
    if missing:
        sys.exit(f"LỖI: không tìm thấy {missing} trong thư mục dữ liệu: {args.data_dir}")
    if os.path.abspath(args.out) == os.path.abspath(args.orig_out):
        sys.exit("LỖI: --out trùng thư mục kết quả gốc; v2 không ghi vào out/.")
    os.makedirs(args.out, exist_ok=True); write_config(args.out, args)
    stages = ["eda", "fit", "eval", "shap", "report"] if args.stage == "all" else [args.stage]
    for s in stages:
        if s == "shap" and not any(m in XGB_MODELS for m in args.models):
            log("bỏ qua stage shap (không có XGBoost)"); continue
        log(f"=== stage: {s} ==="); globals()[f"stage_{s}"](args)


if __name__ == "__main__":
    # Chạy qua module 'run_study_v2' (không phải __main__) để các lớp tự định nghĩa (XGBWeighted, Winsorizer)
    # pickle được sang tiến trình con của joblib và nạp lại được từ cache ở các stage sau.
    sys.path.insert(0, BASE_DIR)
    import run_study_v2
    run_study_v2.main()
