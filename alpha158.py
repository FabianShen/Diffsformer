"""
alpha158.py — Alpha158 因子计算（纯 numpy/pandas）

从 data/ohlcv/{domain}/ohlcv.parquet 读取清洗后的日频 OHLCV，计算 158 个 Alpha158 因子，
落盘 data/alpha158/{domain}/factors.parquet。公式对齐 Qlib `Alpha158DL`。

依赖：market_data.py 先产出 data/ohlcv/。本模块不直接调用 ymm_data_sdk。
本模块只做「数据准备」（因子计算 + 未来收益标签）；归一化（robust z-score）、序列切分
等「数据可训练化」逻辑已移至 diffsformer_data.py，本模块不 import 其它准备层模块。

Factor Composition (158 total):
  K-bar patterns:       9  (KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2)
  Price ratios:         4  (OPEN0, HIGH0, LOW0, VWAP0)
  Rolling × 5 windows: 145  (29 metrics × [5,10,20,30,60])
  Total:               158

Paper: Gao et al. (2024) DiffsFormer
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

# 2. Alpha158 Factor Computation (Pure Numpy/Pandas)
# ============================================================================

def compute_alpha158_factors(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 158 Alpha158 factors from OHLCV data.

    Uses a hybrid approach:
      - K-bar + Price: direct numpy (instant)
      - Simple rolling: per-stock numpy strided views (fast)
      - CORR/CORD: vectorized algebraic formula (fast)
      - BETA/RSQR/RESI: per-stock polyfit (moderate, few params)
    """
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)

    eps = 1e-12
    instruments = df["instrument"].values
    unique_inst = df["instrument"].unique()

    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    v = df["volume"].values.astype(np.float64)
    vw = df["vwap"].values.astype(np.float64)

    n_rows = len(df)
    windows = [5, 10, 20, 30, 60]

    # ── 1. K-bar (9) + Price (4) — direct ──
    result_cols = {}
    result_cols["KMID"]  = (c - o) / o
    result_cols["KLEN"]  = (h - l) / o
    result_cols["KMID2"] = (c - o) / (h - l + eps)
    result_cols["KUP"]   = (h - np.maximum(o, c)) / o
    result_cols["KUP2"]  = (h - np.maximum(o, c)) / (h - l + eps)
    result_cols["KLOW"]  = (np.minimum(o, c) - l) / o
    result_cols["KLOW2"] = (np.minimum(o, c) - l) / (h - l + eps)
    result_cols["KSFT"]  = (2*c - h - l) / o
    result_cols["KSFT2"] = (2*c - h - l) / (h - l + eps)
    result_cols["OPEN0"] = o / c
    result_cols["HIGH0"] = h / c
    result_cols["LOW0"]  = l / c
    result_cols["VWAP0"] = vw / c

    # ── 2. Rolling factors — per-stock numpy (fast path) ──
    rolling_names_all = []
    for w in windows:
        rolling_names_all.extend([
            f"ROC{w}", f"MA{w}", f"STD{w}", f"BETA{w}", f"RSQR{w}", f"RESI{w}",
            f"MAX{w}", f"MIN{w}", f"QTLU{w}", f"QTLD{w}", f"RANK{w}", f"RSV{w}",
            f"IMAX{w}", f"IMIN{w}", f"IMXD{w}", f"CORR{w}", f"CORD{w}",
            f"CNTP{w}", f"CNTN{w}", f"CNTD{w}", f"SUMP{w}", f"SUMN{w}", f"SUMD{w}",
            f"VMA{w}", f"VSTD{w}", f"WVMA{w}", f"VSUMP{w}", f"VSUMN{w}", f"VSUMD{w}",
        ])
    for name in rolling_names_all:
        result_cols[name] = np.full(n_rows, np.nan, dtype=np.float64)

    # Pre-compute for linear regression (same for all stocks)
    x_reg = {w: np.arange(w, dtype=np.float64) for w in windows}
    x_mean = {w: (w-1)/2.0 for w in windows}
    x_demean = {w: x_reg[w] - x_mean[w] for w in windows}
    x_var = {w: (x_demean[w]**2).sum() for w in windows}

    for inst in unique_inst:
        mask = instruments == inst
        idx = np.where(mask)[0]
        n_inst = len(idx)
        if n_inst < 60:
            continue

        ic = c[mask]; ih = h[mask]; il = l[mask]
        iv = v[mask]; ivw = vw[mask]

        for w in windows:
            # Valid positions
            valid_start = w - 1
            tgt = idx[valid_start:]  # indices in global array

            # Rolling views
            rc = _rolling_view(ic, w)
            rh = _rolling_view(ih, w)
            rl = _rolling_view(il, w)
            rv = _rolling_view(iv, w)

            n_windows = len(rc)
            if n_windows == 0:
                continue

            # Slice targets to match
            tgt_use = tgt[:n_windows]

            # --- Simple rolling ---
            result_cols[f"MA{w}"][tgt_use]  = rc.mean(axis=1) / ic[valid_start:valid_start+n_windows]
            result_cols[f"STD{w}"][tgt_use] = rc.std(axis=1, ddof=1) / ic[valid_start:valid_start+n_windows]
            result_cols[f"MAX{w}"][tgt_use] = rh.max(axis=1) / ic[valid_start:valid_start+n_windows]
            result_cols[f"MIN{w}"][tgt_use] = rl.min(axis=1) / ic[valid_start:valid_start+n_windows]
            result_cols[f"QTLU{w}"][tgt_use] = np.quantile(rc, 0.8, axis=1) / ic[valid_start:valid_start+n_windows]
            result_cols[f"QTLD{w}"][tgt_use] = np.quantile(rc, 0.2, axis=1) / ic[valid_start:valid_start+n_windows]

            # ROC: close[t-w] / close[t] — valid from index w
            roc_tgt = idx[w:]
            n_roc = min(n_inst - w, len(roc_tgt))
            result_cols[f"ROC{w}"][roc_tgt[:n_roc]] = ic[:n_inst-w][:n_roc] / ic[w:w+n_roc]

            # RANK
            for i in range(n_windows):
                less = np.sum(rc[i] < rc[i, -1])
                equal = np.sum(rc[i] == rc[i, -1])
                result_cols[f"RANK{w}"][tgt_use[i]] = (less + (equal + 1) / 2) / w

            # RSV
            lo = rl.min(axis=1); hi = rh.max(axis=1)
            result_cols[f"RSV{w}"][tgt_use] = (ic[valid_start:valid_start+n_windows] - lo) / (hi - lo + eps)

            # IMAX / IMIN / IMXD
            result_cols[f"IMAX{w}"][tgt_use] = (rh.argmax(axis=1) + 1) / w
            result_cols[f"IMIN{w}"][tgt_use] = (rl.argmin(axis=1) + 1) / w
            result_cols[f"IMXD{w}"][tgt_use] = (rh.argmax(axis=1) - rl.argmin(axis=1)) / w

            # BETA / RSQR / RESI (linear regression)
            for i in range(n_windows):
                y = rc[i]; xd = x_demean[w]
                slope = np.dot(y - y.mean(), xd) / x_var[w]
                result_cols[f"BETA{w}"][tgt_use[i]] = slope / ic[valid_start+i]
                intercept = y.mean() - slope * x_mean[w]
                y_pred = slope * x_reg[w] + intercept
                ss_res = ((y - y_pred)**2).sum()
                ss_tot = ((y - y.mean())**2).sum()
                result_cols[f"RSQR{w}"][tgt_use[i]] = np.nan if ss_tot <= eps else 1 - ss_res / ss_tot
                result_cols[f"RESI{w}"][tgt_use[i]] = (y[-1] - y_pred[-1]) / ic[valid_start+i]

            # CNTP / CNTN / CNTD
            up = (ic[1:] > ic[:-1]).astype(np.float64)
            dn = (ic[1:] < ic[:-1]).astype(np.float64)
            up_padded = np.concatenate([[np.nan], up])
            dn_padded = np.concatenate([[np.nan], dn])
            ru = _rolling_view(up_padded, w)
            rd = _rolling_view(dn_padded, w)
            if len(ru) >= n_windows:
                result_cols[f"CNTP{w}"][tgt_use] = np.nanmean(ru[-n_windows:], axis=1)
                result_cols[f"CNTN{w}"][tgt_use] = np.nanmean(rd[-n_windows:], axis=1)
                result_cols[f"CNTD{w}"][tgt_use] = (result_cols[f"CNTP{w}"][tgt_use] -
                                                     result_cols[f"CNTN{w}"][tgt_use])

            # SUMP / SUMN / SUMD
            delta_c = np.diff(ic, prepend=ic[0])
            gain = np.maximum(delta_c, 0); loss = np.maximum(-delta_c, 0)
            rg = _rolling_view(gain, w); rl_sum = _rolling_view(loss, w)
            ra = _rolling_view(np.abs(delta_c), w)
            if len(rg) >= n_windows:
                sg, sl, sa = rg[-n_windows:].sum(axis=1), rl_sum[-n_windows:].sum(axis=1), ra[-n_windows:].sum(axis=1)
                result_cols[f"SUMP{w}"][tgt_use] = sg / (sa + eps)
                result_cols[f"SUMN{w}"][tgt_use] = sl / (sa + eps)
                result_cols[f"SUMD{w}"][tgt_use] = (sg - sl) / (sa + eps)

            # VMA / VSTD
            current_volume = iv[valid_start:valid_start+n_windows]
            volume_denom = np.where(current_volume > 0, current_volume, np.nan)
            result_cols[f"VMA{w}"][tgt_use] = rv.mean(axis=1) / volume_denom
            result_cols[f"VSTD{w}"][tgt_use] = rv.std(axis=1, ddof=1) / volume_denom

            # WVMA: Std(|ret|*vol, w) / (Mean(|ret|*vol, w) + eps)
            abs_ret = np.abs(ic[1:] / (ic[:-1] + eps) - 1) * iv[1:]
            rarv = _rolling_view(abs_ret, w)
            n_arv = len(rarv)
            if n_arv > 0:
                # abs_ret starts 1 day later, so target indices need offset
                wvma_tgt = idx[valid_start + 1:valid_start + 1 + n_arv]
                result_cols[f"WVMA{w}"][wvma_tgt] = rarv.std(axis=1, ddof=1) / (rarv.mean(axis=1) + eps)

            # VSUMP / VSUMN / VSUMD
            delta_v = np.diff(iv, prepend=iv[0])
            vgain = np.maximum(delta_v, 0); vloss = np.maximum(-delta_v, 0)
            rvg = _rolling_view(vgain, w); rvl = _rolling_view(vloss, w)
            rva = _rolling_view(np.abs(delta_v), w)
            if len(rvg) >= n_windows:
                sgv = rvg[-n_windows:].sum(axis=1); slv = rvl[-n_windows:].sum(axis=1)
                sav = rva[-n_windows:].sum(axis=1)
                result_cols[f"VSUMP{w}"][tgt_use] = sgv / (sav + eps)
                result_cols[f"VSUMN{w}"][tgt_use] = slv / (sav + eps)
                result_cols[f"VSUMD{w}"][tgt_use] = (sgv - slv) / (sav + eps)

            # --- CORR: vectorized algebraic formula ---
            # corr(x,y) = (n*sum(xy) - sum(x)*sum(y)) / sqrt((n*sum(x²) - sum(x)²)*(n*sum(y²) - sum(y)²))
            logvol = np.log(iv + 1)
            rclose = rc; rlogvol = _rolling_view(logvol, w)
            if n_windows > 0:
                sx = rc.sum(axis=1); sy = rlogvol.sum(axis=1)
                sxy = (rc * rlogvol).sum(axis=1)
                sx2 = (rc * rc).sum(axis=1); sy2 = (rlogvol * rlogvol).sum(axis=1)
                num = w * sxy - sx * sy
                den = np.sqrt(np.maximum((w * sx2 - sx**2) * (w * sy2 - sy**2), 0))
                result_cols[f"CORR{w}"][tgt_use] = np.divide(
                    num[-n_windows:], den[-n_windows:],
                    out=np.full(n_windows, np.nan), where=den[-n_windows:] > eps,
                )

            # --- CORD: Corr(close/Ref(close,1), Log(volume/Ref(volume,1)+1), w) ---
            ret = ic[1:] / (ic[:-1] + eps)
            vol_chg = np.log(iv[1:] / (iv[:-1] + eps) + 1)
            rret = _rolling_view(ret, w)
            rvolchg = _rolling_view(vol_chg, w)
            n_cord = len(rret)
            if n_cord > 0:
                # ret starts 1 day later → target starts at valid_start + 1
                cord_tgt = idx[valid_start + 1:valid_start + 1 + n_cord]
                sx = rret.sum(axis=1); sy = rvolchg.sum(axis=1)
                sxy = (rret * rvolchg).sum(axis=1)
                sx2 = (rret * rret).sum(axis=1); sy2 = (rvolchg * rvolchg).sum(axis=1)
                num = w * sxy - sx * sy
                den = np.sqrt(np.maximum((w * sx2 - sx**2) * (w * sy2 - sy**2), 0))
                result_cols[f"CORD{w}"][cord_tgt] = np.divide(
                    num, den, out=np.full(n_cord, np.nan), where=den > eps,
                )

    # ── Build result DataFrame ──
    # Factor order must match Qlib's Alpha158DL EXACTLY:
    #   9 K-bar → 4 Price → (29 types × 5 windows each, by FACTOR TYPE first)
    # Qlib order: for name in [ROC,MA,...]: for w in [5,10,20,30,60]: f"{name}{w}"
    rolling_names_ordered = [
        "ROC", "MA", "STD", "BETA", "RSQR", "RESI",
        "MAX", "MIN", "QTLU", "QTLD", "RANK", "RSV",
        "IMAX", "IMIN", "IMXD", "CORR", "CORD",
        "CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD",
        "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD",
    ]
    factor_names = (
        ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"] +
        ["OPEN0", "HIGH0", "LOW0", "VWAP0"] +
        [f"{name}{w}" for name in rolling_names_ordered for w in windows]
    )

    # Build all factor columns at once to avoid fragmentation
    factor_data = {}
    for name in factor_names:
        factor_data[name] = result_cols.get(name, np.full(n_rows, np.nan))
    factor_df_only = pd.DataFrame(factor_data, index=df.index)

    # Attach date + instrument, drop rows where w=60 factors are NaN
    result = pd.concat([
        df[["date", "instrument"]].reset_index(drop=True),
        factor_df_only.reset_index(drop=True),
    ], axis=1)

    # Only require non-NaN for the 60-day window factors
    w60_cols = [f"{name}60" for name in ["ROC","MA","STD","BETA","RSQR","RESI",
               "MAX","MIN","QTLU","QTLD","RANK","RSV","IMAX","IMIN","IMXD",
               "CORR","CORD","CNTP","CNTN","CNTD","SUMP","SUMN","SUMD",
               "VMA","VSTD","WVMA","VSUMP","VSUMN","VSUMD"]]
    # Also require K-bar and price factors (should never be NaN)
    required_cols = ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
                     "OPEN0", "HIGH0", "LOW0", "VWAP0"] + w60_cols
    result = result.dropna(subset=required_cols)
    result = result.set_index(["date", "instrument"])

    # Report NaN stats
    nan_counts = {name: result_cols.get(name, np.array([])) for name in factor_names}
    all_nan_cols = [name for name in factor_names
                    if name in result_cols and np.all(np.isnan(result_cols[name]))]
    if all_nan_cols:
        print(f"[Alpha158] WARNING: {len(all_nan_cols)} columns are all-NaN: {all_nan_cols[:5]}...")

    print(f"[Alpha158] Computed {len(factor_names)} factors: {result.shape[0]:,} valid samples")
    return result


def compute_forward_returns(ohlcv: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Return close-to-close forward returns keyed by (date, instrument)."""
    data = ohlcv.sort_values(["instrument", "date"]).copy()
    data["label"] = data.groupby("instrument")["close"].shift(-horizon) / data["close"] - 1
    return data.set_index(["date", "instrument"])["label"].dropna()


def _rolling_view(arr: np.ndarray, window: int) -> np.ndarray:
    """Create a 2D view of rolling windows over a 1D array."""
    if len(arr) < window:
        return np.array([]).reshape(0, window)
    shape = (len(arr) - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


# ============================================================================
# 3. Store factors to data/alpha158
# ============================================================================

@dataclass
class FactorBatch:
    """A batch of factor data in DiffsFormer-compatible format."""
    features: np.ndarray   # (N, lookback, n_factors)
    labels: np.ndarray     # (N,)
    dates: np.ndarray      # (N,)
    instruments: np.ndarray  # (N,)

    def __repr__(self):
        return (f"FactorBatch(n={len(self.labels):,}, "
                f"features={self.features.shape}, "
                f"dates={self.dates[0]}...{self.dates[-1]})")




def compute_and_store_factors(
    domain: str = "target",
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """读取 data/ohlcv/{domain}/ohlcv.parquet，计算 Alpha158，落盘 data/alpha158/{domain}/factors.parquet。"""
    data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
    src_path = data_dir / "ohlcv" / domain / "ohlcv.parquet"
    if not src_path.exists():
        raise FileNotFoundError(f"缺少 {src_path}，请先运行 market_data.build_{domain}_ohlcv()")
    print(f"[alpha158] 读取 {src_path}")
    ohlcv = pd.read_parquet(src_path)
    factors = compute_alpha158_factors(ohlcv)
    out_dir = data_dir / "alpha158" / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    factors.reset_index().to_parquet(out_dir / "factors.parquet", index=False)
    print(f"[alpha158] 已落盘 {out_dir / 'factors.parquet'}（{factors.shape[0]:,} × {factors.shape[1]}）")
    return factors
