import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import average_precision_score, roc_auc_score

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

import matplotlib.pyplot as plt


@dataclass
class Config:
    k: int = 120
    tau: int = 40
    topk: int = 20
    recent_query_days: int = 300
    only_noncritical_query: bool = True
    keep_pct: float = 0.05
    min_keep: int = 300
    max_keep: int = 1200
    dtw_window: int = 10
    stretch_factors: Tuple[float, ...] = (0.9, 1.0, 1.1)
    purge_len: Optional[int] = None

    def __post_init__(self) -> None:
        if self.purge_len is None:
            self.purge_len = self.k + self.tau


def compute_critical_day(pos_conf: pd.Series, neg_conf: pd.Series) -> pd.Series:
    return ((pos_conf > neg_conf) & (pos_conf > 0)) | ((neg_conf > pos_conf) & (neg_conf > 0))


def compute_delta_to_next_critical(critical: np.ndarray) -> np.ndarray:
    n = len(critical)
    next_idx = np.full(n, np.nan)
    next_critical = np.nan
    for i in range(n - 1, -1, -1):
        if critical[i]:
            next_critical = i
        if np.isnan(next_critical):
            next_idx[i] = np.nan
        else:
            next_idx[i] = next_critical - i
    return next_idx


def zscore_window(window: np.ndarray) -> np.ndarray:
    mean = window.mean(axis=0, keepdims=True)
    std = window.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (window - mean) / std


def feature_vector(window: np.ndarray) -> np.ndarray:
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    last = window[-1]
    slope = window[-1] - window[0]
    return np.concatenate([mean, std, last, slope])


def interpolate_window(window: np.ndarray, new_len: int) -> np.ndarray:
    old_len = window.shape[0]
    if new_len == old_len:
        return window.copy()
    x_old = np.linspace(0.0, 1.0, old_len)
    x_new = np.linspace(0.0, 1.0, new_len)
    stretched = np.zeros((new_len, window.shape[1]), dtype=float)
    for dim in range(window.shape[1]):
        stretched[:, dim] = np.interp(x_new, x_old, window[:, dim])
    return stretched


def dtw_distance_sakoe_chiba(x: np.ndarray, y: np.ndarray, window: int) -> float:
    n, m = len(x), len(y)
    w = max(window, abs(n - m))
    inf = np.inf
    dtw = np.full((n + 1, m + 1), inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m, i + w)
        for j in range(j_start, j_end + 1):
            cost = np.sum((x[i - 1] - y[j - 1]) ** 2)
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def min_dtw_with_stretch(query: np.ndarray, candidate: np.ndarray, cfg: Config) -> float:
    best = np.inf
    for factor in cfg.stretch_factors:
        new_len = max(2, int(round(len(candidate) * factor)))
        stretched = interpolate_window(candidate, new_len)
        dist = dtw_distance_sakoe_chiba(query, stretched, cfg.dtw_window)
        if dist < best:
            best = dist
    return best


def select_candidates(
    query_feat: np.ndarray,
    candidate_feats: np.ndarray,
    candidate_indices: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    if len(candidate_indices) == 0:
        return candidate_indices
    diffs = candidate_feats - query_feat
    dists = np.linalg.norm(diffs, axis=1)
    keep_target = int(len(candidate_indices) * cfg.keep_pct)
    keep_target = max(cfg.min_keep, keep_target)
    keep_target = min(cfg.max_keep, keep_target)
    keep_target = min(len(candidate_indices), keep_target)
    keep_idx = np.argpartition(dists, keep_target - 1)[:keep_target]
    return candidate_indices[keep_idx]


def build_candidate_features(data: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    n = len(data)
    end_indices = np.arange(cfg.k - 1, n)
    feats = []
    for idx in end_indices:
        window = data[idx - cfg.k + 1 : idx + 1]
        feats.append(feature_vector(window))
    return end_indices, np.vstack(feats)


def risk_for_query(
    query_idx: int,
    data: np.ndarray,
    end_indices: np.ndarray,
    feats: np.ndarray,
    delta_to_next: np.ndarray,
    cfg: Config,
) -> Tuple[float, int]:
    query_window = data[query_idx - cfg.k + 1 : query_idx + 1]
    query_norm = zscore_window(query_window)
    query_feat = feature_vector(query_window)

    valid_mask = end_indices <= (query_idx - cfg.purge_len)
    candidate_indices = end_indices[valid_mask]
    if len(candidate_indices) == 0:
        return np.nan, 0

    candidate_feats = feats[valid_mask]
    screened_indices = select_candidates(query_feat, candidate_feats, candidate_indices, cfg)

    if len(screened_indices) == 0:
        return np.nan, 0

    dtw_distances = []
    for idx in screened_indices:
        candidate_window = data[idx - cfg.k + 1 : idx + 1]
        candidate_norm = zscore_window(candidate_window)
        dist = min_dtw_with_stretch(query_norm, candidate_norm, cfg)
        dtw_distances.append((idx, dist))

    dtw_distances.sort(key=lambda x: x[1])
    top = dtw_distances[: cfg.topk]
    if not top:
        return np.nan, 0

    top_indices = [idx for idx, _ in top]
    top_deltas = delta_to_next[top_indices]
    valid = ~np.isnan(top_deltas)
    if valid.sum() == 0:
        return np.nan, len(top_indices)
    p_hat = float(np.mean(top_deltas[valid] <= cfg.tau))
    return p_hat, len(top_indices)


def make_clusters(critical_indices: Sequence[int], min_gap: int = 5) -> List[Tuple[int, int]]:
    if not critical_indices:
        return []
    clusters = []
    start = critical_indices[0]
    prev = start
    for idx in critical_indices[1:]:
        if idx - prev <= min_gap:
            prev = idx
            continue
        clusters.append((start, prev))
        start = idx
        prev = idx
    clusters.append((start, prev))
    return clusters


def main() -> None:
    cfg = Config()
    csv_path = os.path.join(os.path.dirname(__file__), "日期对齐.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Date" not in df.columns:
        raise ValueError("输入文件必须包含 Date 列")

    df["Date"] = pd.to_datetime(df["Date"])

    if "pos_conf" not in df.columns or "neg_conf" not in df.columns:
        raise ValueError("输入文件必须包含 pos_conf, neg_conf 列")

    critical_day = compute_critical_day(df["pos_conf"], df["neg_conf"]).astype(int)
    delta_to_next = compute_delta_to_next_critical(critical_day.to_numpy())

    feature_cols = [col for col in df.columns if col not in {"Date"}]
    data = df[feature_cols].to_numpy(dtype=float)

    end_indices, feats = build_candidate_features(data, cfg)

    n = len(df)
    start_idx = max(cfg.k - 1, n - cfg.recent_query_days)
    query_indices = list(range(start_idx, n))
    if cfg.only_noncritical_query:
        query_indices = [idx for idx in query_indices if critical_day.iloc[idx] == 0]

    results = []
    for idx in query_indices:
        p_hat, used = risk_for_query(idx, data, end_indices, feats, delta_to_next, cfg)
        results.append({"Date": df.loc[idx, "Date"], "P_hat": p_hat, "n_neighbors": used})

    out_dir = os.path.join(os.path.dirname(__file__), "runs_lppls_risk")
    os.makedirs(out_dir, exist_ok=True)

    risk_df = pd.DataFrame(results)
    risk_df.to_csv(os.path.join(out_dir, "risk_scores.csv"), index=False)

    eval_rows = {
        "n_samples": len(risk_df),
        "base_rate": np.nan,
        "AUC": np.nan,
        "PR_AUC": np.nan,
    }

    if len(risk_df) > 0:
        query_idx_map = {df.loc[idx, "Date"]: idx for idx in query_indices}
        labels = []
        preds = []
        for _, row in risk_df.iterrows():
            idx = query_idx_map.get(row["Date"])
            if idx is None:
                continue
            label = float(delta_to_next[idx] <= cfg.tau) if not np.isnan(delta_to_next[idx]) else np.nan
            labels.append(label)
            preds.append(row["P_hat"])

        labels = np.array(labels, dtype=float)
        preds = np.array(preds, dtype=float)
        valid = ~np.isnan(labels) & ~np.isnan(preds)
        if valid.any():
            eval_rows["base_rate"] = float(np.mean(labels[valid]))
            if SKLEARN_AVAILABLE:
                try:
                    eval_rows["AUC"] = float(roc_auc_score(labels[valid], preds[valid]))
                    eval_rows["PR_AUC"] = float(average_precision_score(labels[valid], preds[valid]))
                except Exception:
                    eval_rows["AUC"] = np.nan
                    eval_rows["PR_AUC"] = np.nan

    eval_df = pd.DataFrame([eval_rows])
    eval_df.to_csv(os.path.join(out_dir, "eval_day_level.csv"), index=False)

    plt.figure(figsize=(12, 4))
    plt.plot(risk_df["Date"], risk_df["P_hat"], label="P_hat")
    critical_indices = np.where(critical_day.to_numpy() == 1)[0].tolist()
    clusters = make_clusters(critical_indices, min_gap=5)
    for start, end in clusters:
        plt.axvspan(df.loc[start, "Date"], df.loc[end, "Date"], color="red", alpha=0.1)
    plt.title("LPPLS Analog Risk (P_hat)")
    plt.xlabel("Date")
    plt.ylabel("Risk Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Fig_risk_curve.png"), dpi=150)


if __name__ == "__main__":
    main()
