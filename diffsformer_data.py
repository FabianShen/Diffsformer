"""YMM data preparation and restartable array caches for DiffsFormer."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

# 数据源：本模块只读落盘数据（data/merged、data/ohlcv、data/industry、data/benchmark），
# 不 import 任何准备层模块（market_data / alpha158 / labels），本模块零 ymm_data_sdk 依赖。
# 「数据可训练化」逻辑（robust z-score 归一化 + 序列切分 + 切集 + 行业编码 + 基准收益）全部在本模块内。
DEFAULT_DATA_DIR = Path(__file__).parent / "data"


@dataclass
class TargetData:
    train_features: np.ndarray
    train_labels: np.ndarray
    train_industries: np.ndarray
    val_features: np.ndarray
    val_labels: np.ndarray
    test_features: np.ndarray
    test_labels: np.ndarray
    test_dates: np.ndarray
    test_instruments: np.ndarray
    test_returns: np.ndarray
    factor_names: np.ndarray
    benchmark_dates: np.ndarray
    benchmark_returns: np.ndarray
    scaler_median: np.ndarray
    scaler_scale: np.ndarray
    label_low: float
    label_high: float
    num_industries: int
    fingerprint: str
    metadata: dict[str, Any]


@dataclass
class SourceData:
    features: np.ndarray
    labels: np.ndarray
    industries: np.ndarray
    fingerprint: str
    metadata: dict[str, Any]


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _read_manifest(folder: Path) -> dict[str, Any] | None:
    path = folder / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(folder: Path, manifest: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    temp = folder / "manifest.json.tmp"
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(folder / "manifest.json")


def _load_array(folder: Path, name: str) -> np.ndarray:
    return np.load(folder / f"{name}.npy", mmap_mode="r+", allow_pickle=False)


def fit_robust_zscore_factors(
    factor_df: pd.DataFrame,
    fit_end: str | pd.Timestamp,
    fit_index: Optional[pd.MultiIndex] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Fit the median/MAD scaler on the training period (anti-lookahead: only dates <= fit_end
    and instruments in fit_index)."""
    clean = factor_df.replace([np.inf, -np.inf], np.nan).astype(float)
    dates = clean.index.get_level_values("date")
    fit_mask = dates <= pd.Timestamp(fit_end)
    if fit_index is not None:
        fit_mask &= clean.index.isin(fit_index)
    train = clean.loc[fit_mask]
    if train.empty:
        raise ValueError("No factor rows available in the normalization fit period")

    median = train.median()
    scale = (train - median).abs().median() * 1.4826
    scale = scale.where(scale > 1e-12, 1.0)
    return median, scale


def apply_robust_zscore_factors(
    factor_df: pd.DataFrame,
    median: pd.Series,
    scale: pd.Series,
    clip: float = 3.0,
) -> pd.DataFrame:
    """Apply a previously fitted median/MAD scaler to Alpha158 factors."""
    clean = factor_df.replace([np.inf, -np.inf], np.nan).astype(float)
    missing = clean.columns.difference(median.index)
    if len(missing):
        raise ValueError(f"Scaler is missing {len(missing)} factor columns")
    normalized = ((clean - median[clean.columns]) / scale[clean.columns]).clip(-clip, clip).fillna(0.0)
    return normalized.astype(np.float32)


def build_factor_sequences(
    factor_df: pd.DataFrame,
    labels: pd.Series,
    lookback: int = 8,
    membership: Optional[pd.MultiIndex] = None,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (N, lookback, n_factors) sliding window sequences from factor data.

    Args:
        factor_df: DataFrame with MultiIndex (date, instrument), n factor columns
        labels: keyed Series with MultiIndex (date, instrument)
        lookback: number of past days to include in each sequence
        membership: optional point-in-time universe keys
        stride: keep every ``stride``-th sequence for each instrument
    Returns:
        features: (N, lookback, n_factors)
        labels_arr: (N,)
        dates_arr: (N,) — date of each sample
        instruments_arr: (N,) — instrument of each sample
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    n_factors = factor_df.shape[1]
    factor_arr = factor_df.values  # (n_samples, n_factors)
    index_vals = factor_df.index

    membership_set = set(membership) if membership is not None else None

    sequences = []
    labels_list = []
    dates_list = []
    insts_list = []

    # Group by instrument
    instruments = index_vals.get_level_values("instrument")
    dates = index_vals.get_level_values("date")
    unique_insts = instruments.unique()

    for inst in unique_insts:
        inst_mask = instruments == inst
        inst_dates = dates[inst_mask]
        inst_indices = np.where(inst_mask)[0]

        for t in range(lookback - 1, len(inst_indices), stride):
            sample_key = (inst_dates[t], inst)
            if membership_set is not None and sample_key not in membership_set:
                continue
            if sample_key not in labels.index:
                continue

            seq_indices = inst_indices[t - lookback + 1:t + 1]
            seq = factor_arr[seq_indices]  # (lookback, n_factors)
            label_val = labels.loc[sample_key]
            if not np.isfinite(seq).all() or not np.isfinite(label_val):
                continue

            sequences.append(seq)
            labels_list.append(label_val)
            dates_list.append(sample_key[0])
            insts_list.append(inst)

    return (
        np.array(sequences, dtype=np.float32),
        np.array(labels_list, dtype=np.float32).flatten(),
        np.array(dates_list),
        np.array(insts_list),
    )


def _industry_vocabulary(date: str, data_dir: Path) -> tuple[dict[str, int], int]:
    path = data_dir / "industry" / date / "mapping.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 {path}，请先运行 `python market_data.py industry --date {date}`"
        )
    mapping = pd.read_parquet(path)
    codes = sorted(mapping["first_industry_code"].dropna().astype(str).unique())
    vocabulary = {code: index + 1 for index, code in enumerate(codes)}
    return vocabulary, len(vocabulary) + 1  # zero is unknown


def _instrument_industries(
    date: str, vocabulary: dict[str, int], data_dir: Path,
) -> dict[str, int]:
    path = data_dir / "industry" / date / "instrument_industry.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 {path}，请先运行 `python market_data.py industry --date {date}`"
        )
    table = pd.read_parquet(path)
    return {
        str(inst): vocabulary.get(str(code), 0)
        for inst, code in zip(table["instrument"], table["first_industry_code"])
    }


def _label_safe_cutoff(
    sessions: pd.DatetimeIndex, next_period_start: str, horizon: int,
) -> pd.Timestamp:
    prior = sessions[sessions < pd.Timestamp(next_period_start)]
    if len(prior) <= horizon:
        raise ValueError("Not enough sessions to purge the split boundary")
    return pd.Timestamp(prior[-(horizon + 1)])


def prepare_target_data(config: dict[str, Any], cache_root: str | Path) -> TargetData:
    """Prepare the exact extended CSI300 benchmark and cache its arrays."""
    cache_root = Path(cache_root)
    cache_payload = {
        "version": 3,
        "kind": "target_csi300",
        "data": {key: config[key] for key in (
            "start_date", "fetch_end_date", "train_end", "valid_start",
            "valid_end", "test_start", "test_end", "lookback",
            "label_horizon", "purge_sessions",
        )},
    }
    fingerprint = _fingerprint(cache_payload)
    folder = cache_root / f"target_{fingerprint}"
    manifest = _read_manifest(folder)
    if manifest and manifest.get("complete") and manifest.get("fingerprint") == fingerprint:
        print(f"[Data] Loading target cache {folder}")
        return _load_target(folder, manifest)

    folder.mkdir(parents=True, exist_ok=True)
    data_dir = Path(config.get("data_dir", DEFAULT_DATA_DIR))
    merged_path = data_dir / "merged" / "target" / "merged.parquet"
    if not merged_path.exists():
        raise FileNotFoundError(f"缺少 {merged_path}，请先运行 main.py 生成 data/merged/target")
    merged = pd.read_parquet(merged_path)
    merged["date"] = pd.to_datetime(merged["date"])
    factor_cols = [c for c in merged.columns if c not in ("date", "instrument", "label")]
    raw_factors = merged.set_index(["date", "instrument"])[factor_cols]

    membership_frame = pd.read_parquet(data_dir / "ohlcv" / "target" / "membership.parquet")
    membership_frame["date"] = pd.to_datetime(membership_frame["date"])
    membership = pd.MultiIndex.from_frame(membership_frame[["date", "instrument"]])

    ohlcv = pd.read_parquet(data_dir / "ohlcv" / "target" / "ohlcv.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    median, scale = fit_robust_zscore_factors(
        raw_factors, config["train_end"], fit_index=membership
    )
    factors = apply_robust_zscore_factors(raw_factors, median, scale)
    factor_names = factors.columns.to_numpy(dtype=str)
    # 标签已由 labels.py 按 horizon=5 预计算并过滤后存入 merged.parquet；
    # config["label_horizon"] 仅进入缓存指纹，实际口径以 merged 为准（须与 labels.py 一致）。
    label_series = merged.set_index(["date", "instrument"])["label"]
    features, labels, dates, instruments = build_factor_sequences(
        factors, label_series, lookback=config["lookback"], membership=membership
    )

    date_index = pd.DatetimeIndex(pd.to_datetime(dates))
    sessions = pd.DatetimeIndex(pd.unique(ohlcv["date"])).sort_values()
    train_cutoff = min(
        pd.Timestamp(config["train_end"]),
        _label_safe_cutoff(sessions, config["valid_start"], config["purge_sessions"]),
    )
    valid_cutoff = min(
        pd.Timestamp(config["valid_end"]),
        _label_safe_cutoff(sessions, config["test_start"], config["purge_sessions"]),
    )
    train_mask = date_index <= train_cutoff
    val_mask = (
        (date_index >= pd.Timestamp(config["valid_start"]))
        & (date_index <= valid_cutoff)
    )
    test_mask = (
        (date_index >= pd.Timestamp(config["test_start"]))
        & (date_index <= pd.Timestamp(config["test_end"]))
    )
    label_low, label_high = np.quantile(labels[train_mask], [0.01, 0.99])
    train_mask &= (labels >= label_low) & (labels <= label_high)

    vocabulary, num_industries = _industry_vocabulary(config["train_end"], data_dir)
    train_instruments = instruments[train_mask]
    industry_map = _instrument_industries(config["train_end"], vocabulary, data_dir)
    train_industries = np.fromiter(
        (industry_map.get(str(inst), 0) for inst in train_instruments),
        dtype=np.int64, count=len(train_instruments),
    )

    bars = ohlcv.sort_values(["instrument", "date"]).copy()
    bars["realized_return"] = (
        bars.groupby("instrument")["open"].shift(-2)
        / bars.groupby("instrument")["open"].shift(-1) - 1
    )
    test_dates = dates[test_mask].astype("datetime64[ns]")
    test_instruments = instruments[test_mask].astype(str)
    test_keys = pd.MultiIndex.from_arrays(
        [pd.to_datetime(test_dates), test_instruments], names=["date", "instrument"]
    )
    realized = bars.set_index(["date", "instrument"])["realized_return"]
    test_returns = realized.reindex(test_keys).to_numpy(dtype=np.float64)

    benchmark_path = data_dir / "benchmark" / "000300.XSHG.parquet"
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"缺少 {benchmark_path}，请先运行 `python market_data.py benchmark "
            f"--start {config['start_date']} --end {config['fetch_end_date']}`"
        )
    index_bars = pd.read_parquet(benchmark_path)
    index_bars["benchmark_return"] = (
        index_bars["open"].shift(-2) / index_bars["open"].shift(-1) - 1
    )

    arrays = {
        "train_features": features[train_mask].astype(np.float32, copy=False),
        "train_labels": labels[train_mask].astype(np.float32, copy=False),
        "train_industries": train_industries,
        "val_features": features[val_mask].astype(np.float32, copy=False),
        "val_labels": labels[val_mask].astype(np.float32, copy=False),
        "test_features": features[test_mask].astype(np.float32, copy=False),
        "test_labels": labels[test_mask].astype(np.float32, copy=False),
        "test_dates": test_dates,
        "test_instruments": test_instruments,
        "test_returns": test_returns,
        "factor_names": factor_names,
        "benchmark_dates": index_bars["date"].to_numpy(dtype="datetime64[ns]"),
        "benchmark_returns": index_bars["benchmark_return"].to_numpy(dtype=np.float64),
        "scaler_median": median[factor_names].to_numpy(dtype=np.float64),
        "scaler_scale": scale[factor_names].to_numpy(dtype=np.float64),
    }
    for name, array in arrays.items():
        np.save(folder / f"{name}.npy", array, allow_pickle=False)

    metadata = {
        "train_cutoff": str(train_cutoff.date()),
        "valid_cutoff": str(valid_cutoff.date()),
        "train_shape": list(arrays["train_features"].shape),
        "val_shape": list(arrays["val_features"].shape),
        "test_shape": list(arrays["test_features"].shape),
        "test_first": str(pd.Timestamp(test_dates.min()).date()),
        "test_last": str(pd.Timestamp(test_dates.max()).date()),
    }
    manifest = {
        **cache_payload, "fingerprint": fingerprint, "complete": True,
        "label_low": float(label_low), "label_high": float(label_high),
        "num_industries": num_industries, "metadata": metadata,
    }
    _write_manifest(folder, manifest)
    del features, labels, factors, raw_factors, ohlcv, bars, merged, label_series, membership, membership_frame
    gc.collect()
    return _load_target(folder, manifest)


def _load_target(folder: Path, manifest: dict[str, Any]) -> TargetData:
    return TargetData(
        train_features=_load_array(folder, "train_features"),
        train_labels=_load_array(folder, "train_labels"),
        train_industries=_load_array(folder, "train_industries"),
        val_features=_load_array(folder, "val_features"),
        val_labels=_load_array(folder, "val_labels"),
        test_features=_load_array(folder, "test_features"),
        test_labels=_load_array(folder, "test_labels"),
        test_dates=_load_array(folder, "test_dates"),
        test_instruments=_load_array(folder, "test_instruments"),
        test_returns=_load_array(folder, "test_returns"),
        factor_names=_load_array(folder, "factor_names"),
        benchmark_dates=_load_array(folder, "benchmark_dates"),
        benchmark_returns=_load_array(folder, "benchmark_returns"),
        scaler_median=_load_array(folder, "scaler_median"),
        scaler_scale=_load_array(folder, "scaler_scale"),
        label_low=float(manifest["label_low"]),
        label_high=float(manifest["label_high"]),
        num_industries=int(manifest["num_industries"]),
        fingerprint=manifest["fingerprint"],
        metadata=manifest["metadata"],
    )


def _stable_rng(seed: int, instrument: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{instrument}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def prepare_source_data(
    config: dict[str, Any], target: TargetData, cache_root: str | Path,
) -> SourceData:
    """Build a balanced all-A-share source sample with chunk-level resume."""
    required = (
        "source_start", "source_end", "lookback", "label_horizon",
        "source_stride", "source_max_samples", "source_chunk_size", "seed",
    )
    payload = {
        "version": 3, "kind": "source_all_a",
        "target_fingerprint": target.fingerprint,
        "data": {key: config[key] for key in required},
    }
    fingerprint = _fingerprint(payload)
    folder = Path(cache_root) / f"source_{fingerprint}"
    manifest = _read_manifest(folder)
    if manifest and manifest.get("complete") and manifest.get("fingerprint") == fingerprint:
        print(f"[Data] Loading source cache {folder}")
        return _load_source(folder, manifest)

    folder.mkdir(parents=True, exist_ok=True)
    data_dir = Path(config.get("data_dir", DEFAULT_DATA_DIR))
    merged_path = data_dir / "merged" / "source" / "merged.parquet"
    if not merged_path.exists():
        raise FileNotFoundError(f"缺少 {merged_path}，请先运行 main.py 生成 data/merged/source")
    merged = pd.read_parquet(merged_path)
    merged["date"] = pd.to_datetime(merged["date"])
    factor_cols = [c for c in merged.columns if c not in ("date", "instrument", "label")]
    merged = merged.set_index(["date", "instrument"]).sort_index()
    raw_all = merged[factor_cols]
    label_all = merged["label"]

    instruments = np.asarray(pd.unique(merged.index.get_level_values("instrument")))
    rng = np.random.default_rng(config["seed"])
    instruments = instruments[rng.permutation(len(instruments))]
    max_samples = int(config["source_max_samples"])
    quota = math.ceil(max_samples / len(instruments))
    vocabulary, num_industries = _industry_vocabulary(config["source_end"], data_dir)
    industry_map = _instrument_industries(config["source_end"], vocabulary, data_dir)
    median = pd.Series(target.scaler_median, index=target.factor_names.astype(str))
    scale = pd.Series(target.scaler_scale, index=target.factor_names.astype(str))

    feature_path = folder / "features.npy"
    label_path = folder / "labels.npy"
    industry_path = folder / "industries.npy"
    if manifest and manifest.get("fingerprint") == fingerprint:
        next_index, count = int(manifest["next_index"]), int(manifest["count"])
        feature_store = np.lib.format.open_memmap(feature_path, mode="r+")
        label_store = np.lib.format.open_memmap(label_path, mode="r+")
        industry_store = np.lib.format.open_memmap(industry_path, mode="r+")
        print(f"[Data] Resuming source cache at instrument {next_index}, samples {count:,}")
    else:
        next_index, count = 0, 0
        feature_store = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float32,
            shape=(max_samples, config["lookback"], len(target.factor_names)),
        )
        label_store = np.lib.format.open_memmap(
            label_path, mode="w+", dtype=np.float32, shape=(max_samples,),
        )
        industry_store = np.lib.format.open_memmap(
            industry_path, mode="w+", dtype=np.int64, shape=(max_samples,),
        )

    chunk_size = int(config["source_chunk_size"])
    for start in range(next_index, len(instruments), chunk_size):
        chunk = instruments[start:start + chunk_size]
        print(
            f"[Data] Source instruments {start + 1}-{start + len(chunk)}/"
            f"{len(instruments)} | samples {count:,}/{max_samples:,}"
        )
        raw = raw_all.loc[raw_all.index.get_level_values("instrument").isin(chunk)]
        factors = apply_robust_zscore_factors(raw, median, scale)
        labels = label_all.loc[factors.index]
        x, y, dates, insts = build_factor_sequences(
            factors, labels, lookback=config["lookback"],
            stride=config["source_stride"],
        )
        valid = (
            (pd.to_datetime(dates) <= pd.Timestamp(target.metadata["train_cutoff"]))
            & (y >= target.label_low) & (y <= target.label_high)
        )
        x, y, insts = x[valid], y[valid], insts[valid].astype(str)
        chosen = []
        for instrument in pd.unique(insts):
            indices = np.flatnonzero(insts == instrument)
            if len(indices) > quota:
                indices = _stable_rng(config["seed"], instrument).choice(
                    indices, quota, replace=False
                )
            chosen.append(indices)
        chosen_idx = np.concatenate(chosen) if chosen else np.empty(0, dtype=int)
        if len(chosen_idx):
            chosen_idx = chosen_idx[rng.permutation(len(chosen_idx))]
            chosen_idx = chosen_idx[:max_samples - count]
            selected_insts = insts[chosen_idx]
            end = count + len(chosen_idx)
            feature_store[count:end] = x[chosen_idx]
            label_store[count:end] = y[chosen_idx]
            industry_store[count:end] = np.fromiter(
                (industry_map.get(inst, 0) for inst in selected_insts),
                dtype=np.int64, count=len(selected_insts),
            )
            count = end
            feature_store.flush(); label_store.flush(); industry_store.flush()
        next_index = start + len(chunk)
        manifest = {
            **payload, "fingerprint": fingerprint, "complete": False,
            "next_index": next_index, "count": count,
            "num_instruments": len(instruments), "num_industries": num_industries,
        }
        _write_manifest(folder, manifest)
        del raw, factors, labels, x, y, dates, insts
        gc.collect()
        if count >= max_samples:
            break

    if count == 0:
        raise RuntimeError("No all-A-share source sequences were produced")
    manifest.update({"complete": True, "count": count})
    _write_manifest(folder, manifest)
    del feature_store, label_store, industry_store
    gc.collect()
    return _load_source(folder, manifest)


def _load_source(folder: Path, manifest: dict[str, Any]) -> SourceData:
    count = int(manifest["count"])
    return SourceData(
        features=_load_array(folder, "features")[:count],
        labels=_load_array(folder, "labels")[:count],
        industries=_load_array(folder, "industries")[:count],
        fingerprint=manifest["fingerprint"],
        metadata={
            "count": count,
            "num_instruments": int(manifest["num_instruments"]),
            "num_industries": int(manifest["num_industries"]),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# 独立运行入口（python diffsformer_data.py）：只做「数据可训练化」并产出缓存张量，
# 不触发任何模型训练。生成物写入 cache_root/target_{指纹}/ 与 cache_root/source_{指纹}/。
# 后续 run.ipynb / train_pipeline.py 调用 prepare_target_data / prepare_source_data
# 时，若 cache_root 一致，会命中缓存直接加载（跳过重算）。
# ──────────────────────────────────────────────────────────────────────────────
SMOKE_SEED = 42
SMOKE_DATA_CONFIG: dict[str, Any] = {
    "start_date": "2023-09-01",
    "fetch_end_date": "2024-02-29",
    "train_end": "2024-01-31",
    "valid_start": "2024-02-01",
    "valid_end": "2024-02-14",
    "test_start": "2024-02-19",
    "test_end": "2024-02-22",
    "lookback": 8,
    "label_horizon": 5,
    "purge_sessions": 5,
}
SMOKE_SOURCE_CONFIG: dict[str, Any] = {
    "source_start": "2023-09-01",
    "source_end": "2024-01-31",
    "lookback": 8,
    "label_horizon": 5,
    "source_stride": 10,
    "source_max_samples": 100_000,
    "source_chunk_size": 64,
    "seed": SMOKE_SEED,
}
DEFAULT_CACHE_ROOT = Path(__file__).parent / "cache" / "diffsformer_smoke"


def main(argv: Optional[list[str]] = None) -> None:
    """处理层入口：目标域 → 源域，产出可训练张量缓存（不训练模型）。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="DiffsFormer 处理层入口：只做数据可训练化并产出缓存张量（不训练模型）"
    )
    parser.add_argument(
        "--cache-root", default=str(DEFAULT_CACHE_ROOT),
        help="缓存根目录（默认 cache/diffsformer_smoke；想验证重算路径就换一个空目录）",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="落盘数据目录（默认 data/）",
    )
    args = parser.parse_args(argv)

    target_config = dict(SMOKE_DATA_CONFIG)
    source_config = dict(SMOKE_SOURCE_CONFIG)
    if args.data_dir:
        target_config["data_dir"] = args.data_dir
        source_config["data_dir"] = args.data_dir

    print("[diffsformer_data] 处理层入口启动（只做数据可训练化，不训练模型）")
    print(f"[diffsformer_data] cache_root = {Path(args.cache_root).resolve()}")
    target = prepare_target_data(target_config, args.cache_root)
    source = prepare_source_data(source_config, target, args.cache_root)

    print("\n[diffsformer_data] 完成，产出缓存张量：")
    print(
        f"  target train {target.train_features.shape} "
        f"/ val {target.val_features.shape} "
        f"/ test {target.test_features.shape}，num_industries={target.num_industries}"
    )
    print(f"  target 缓存目录 {Path(args.cache_root) / f'target_{target.fingerprint}'}")
    print(
        f"  source 样本 {int(source.features.shape[0]):,}，"
        f"instruments={source.metadata['num_instruments']}，"
        f"industries={source.metadata['num_industries']}"
    )
    print(f"  source 缓存目录 {Path(args.cache_root) / f'source_{source.fingerprint}'}")


if __name__ == "__main__":
    main()
