"""main_data.py — 数据准备层唯一入口：一次性准备 DiffsFormer 所需全部数据

一次跑通全部准备步骤，产出 diffsformer_data 消费的全部落盘数据：
  OHLCV → industry → benchmark → Alpha158 因子 → 5 日收益标签 → 合并(merged)

本模块是「数据准备」的唯一编排入口，不调用 diffsformer_data（处理层）。
market_data / alpha158 / labels 只作为库被本模块调用，不再各自独立运行。

用法：
    python main_data.py --start 2023-09-01 --end 2024-02-29 --industry-date 2024-01-31
    python main_data.py --start 2023-09-01 --end 2024-02-29 --industry-date 2024-01-31 --force
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_data import (
    build_benchmark,
    build_industry_data,
    build_source_ohlcv,
    build_target_ohlcv,
)
from alpha158 import compute_and_store_factors
from labels import compute_and_store_labels

DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def merge_factors_labels(domain: str = "target", data_dir: str | Path | None = None) -> pd.DataFrame:
    """加载 data/alpha158 与 data/labels，按 (date, instrument) 内连接合并。

    返回列: date, instrument, [158 因子], label
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    factors = pd.read_parquet(data_dir / "alpha158" / domain / "factors.parquet")
    labels = pd.read_parquet(data_dir / "labels" / domain / "labels.parquet")
    return factors.merge(labels, on=["date", "instrument"], how="inner")


def run_data_pipeline(
    start_date: str,
    end_date: str,
    industry_date: str,
    data_dir: str | Path | None = None,
    force: bool = False,
) -> None:
    """一次性准备全部数据（准备层入口）。

    顺序：OHLCV → industry → benchmark → factors → labels → merged。
    各步命中缓存即跳过（force=True 时重建 OHLCV/industry/benchmark）。
    """
    if not start_date or not end_date or not industry_date:
        raise ValueError("需提供 --start --end --industry-date（YYYY-MM-DD）")
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    # 1. OHLCV（目标域 + 源域；industry 依赖其股票池）
    build_target_ohlcv(start_date, end_date, adjust_type="post", force=force)
    build_source_ohlcv(start_date, end_date, adjust_type="post", force=force)

    # 2. 行业映射（CITIC 一级，股票池 = 源域 OHLCV ∪ 目标域 membership）
    build_industry_data(industry_date, out_dir=data_dir, force=force)

    # 3. 基准指数（CSI300 前复权）
    build_benchmark(start_date, end_date, out_dir=data_dir, force=force)

    # 4-6. 因子 → 标签 → 合并（目标域 + 源域）
    for domain in ("target", "source"):
        compute_and_store_factors(domain, data_dir=data_dir)
        compute_and_store_labels(domain, data_dir=data_dir)
        merged = merge_factors_labels(domain, data_dir=data_dir)
        out_dir = data_dir / "merged" / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out_dir / "merged.parquet", index=False)
        n_factor = merged.shape[1] - 3  # 除去 date, instrument, label
        print(f"[main_data] 已落盘 {out_dir / 'merged.parquet'}（{merged.shape[0]:,} × {n_factor} 因子 + 1 标签）")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DiffsFormer 数据准备层唯一入口：ohlcv -> industry -> benchmark -> alpha158 -> labels -> merged"
    )
    parser.add_argument("--start", required=True, help="开始日期 YYYY-MM-DD（OHLCV 与基准区间起点）")
    parser.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD（OHLCV 与基准区间终点 = fetch_end_date）")
    parser.add_argument("--industry-date", required=True, help="行业快照日期 YYYY-MM-DD（= train_end = source_end）")
    parser.add_argument("--force", action="store_true", help="强制重建 OHLCV/industry/benchmark（忽略缓存）")
    args = parser.parse_args()
    run_data_pipeline(
        start_date=args.start,
        end_date=args.end,
        industry_date=args.industry_date,
        force=args.force,
    )
