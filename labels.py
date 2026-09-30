"""labels.py — 5 日未来收益标签（预测目标）→ data/labels/{domain}/labels.parquet

标签定义（论文 Eq.(1)）：
    RR(i) = (Cprice_{t+i} - Cprice_t) / Cprice_t = close[t+i]/close[t] - 1,   i = 5

逐日滚动计算（每个交易日、每只股票一个标签，相邻标签的收益窗口重叠 4 天），**不做降采样**。

极端标签过滤（对应 notes.md 管线第 6 步 "Drop extreme labels (limit-up/limit-down filtering)"）：
  - 涨跌停：剔除以入场日 close 触及涨停/跌停价的样本（涨停无法买入、跌停伴随重大风险，标签不可实现）
  - 次新股：剔除上市不足 `min_list_days` 自然日的样本（防无涨跌幅限制首日/次新剧烈波动）
阈值口径为研究者假设（论文未给具体阈值），`min_list_days` 可配置。

注意：重叠窗口意味着标签存在序列相关，后续划分 train/val/test 时不能按行随机切分，需按时间切分。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha158 import compute_forward_returns

DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def _filter_extreme_labels(
    labels: pd.DataFrame,
    limit_status: pd.DataFrame,
    horizon: int = 5,
    min_list_days: int = 60,
) -> tuple[pd.DataFrame, dict]:
    """按 (date, instrument) 合并涨跌停/上市信息，剔除极端标签。

    剔除条件（任一命中即剔除）：
      - 持仓窗口 [t, t+horizon] 内任一日触及涨/跌停（入场无法买入、出场无法卖出、中途触板为病态 streak）
      - 次新股：上市不足 min_list_days 自然日

    labels 列: date, instrument, label
    limit_status 列: date, instrument, is_limit_up, is_limit_down, listed_date
    返回: (过滤后 labels, 剔除统计)
    """
    ls = limit_status.sort_values(["instrument", "date"]).copy()
    ls["is_limit"] = ls["is_limit_up"] | ls["is_limit_down"]
    # 组内向后滑动 horizon 天，标记 [t, t+horizon] 任一日触板（含当日 t）
    any_limit = ls["is_limit"].copy()
    for k in range(1, horizon + 1):
        any_limit = any_limit | ls.groupby("instrument")["is_limit"].shift(-k).eq(True)
    ls["any_limit_window"] = any_limit

    m = labels.merge(
        ls[["date", "instrument", "any_limit_window", "listed_date"]],
        on=["date", "instrument"], how="left",
    )
    touched_limit = m["any_limit_window"].eq(True)
    listing_age = (m["date"] - m["listed_date"]).dt.days
    is_new = listing_age < min_list_days
    keep = ~(touched_limit | is_new)
    stats = {
        "n_touched_limit": int(touched_limit.sum()),
        "n_new_listing": int(is_new.sum()),
        "n_dropped": int((~keep).sum()),
        "n_kept": int(keep.sum()),
    }
    return m.loc[keep, ["date", "instrument", "label"]].reset_index(drop=True), stats


def compute_and_store_labels(
    domain: str = "target",
    horizon: int = 5,
    min_list_days: int = 60,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """读取 data/ohlcv/{domain}/(ohlcv, limit_status)，算 5 日标签并过滤极端样本，落盘 data/labels/{domain}/labels.parquet。

    返回列: date, instrument, label
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    src = data_dir / "ohlcv" / domain / "ohlcv.parquet"
    limit_src = data_dir / "ohlcv" / domain / "limit_status.parquet"
    if not src.exists():
        raise FileNotFoundError(f"缺少 {src}，请先运行 market_data.build_{domain}_ohlcv()")
    if not limit_src.exists():
        raise FileNotFoundError(f"缺少 {limit_src}，请先运行 market_data.build_{domain}_ohlcv()")

    print(f"[labels] 读取 {src}")
    ohlcv = pd.read_parquet(src)
    labels = compute_forward_returns(ohlcv, horizon=horizon).rename("label").reset_index()
    limit_status = pd.read_parquet(limit_src)

    out, stats = _filter_extreme_labels(labels, limit_status, horizon=horizon, min_list_days=min_list_days)

    out_dir = data_dir / "labels" / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "labels.parquet", index=False)
    print(
        f"[labels] 原始 {len(labels):,} 标签 → 剔触板 {stats['n_touched_limit']:,}、"
        f"次新股 {stats['n_new_listing']:,} → 保留 {stats['n_kept']:,}"
    )
    print(f"[labels] 已落盘 {out_dir / 'labels.parquet'}（{len(out):,} 行，horizon={horizon}）")
    return out
