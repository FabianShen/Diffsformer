"""market_data.py — 数据获取与清洗（ymm_data_sdk 0.9.23）

单点数据源：所有行情经此模块取数，一次取存到 data/ohlcv/，后续只读本地缓存。
由 main_data.py 统一编排调用（本模块不独立运行）。
口径（与论文对齐，已确认）：
  - 复权：后复权 adjust_type="post"（比值型因子/标签/回测收益均正确，历史可复现、无前视）
  - 停牌：skip_suspended=True 剔除停牌日
  - ST：逐日 point-in-time 剔除（is_st_stock）
  - 成分股：index_components 逐日 point-in-time（无幸存者偏差）
  - 目标域 = CSI300(000300.XSHG) ∪ CSI800(000906.XSHG)；源域 = 全 A 股(all_instruments type="CS")

落盘契约（parquet + manifest.json，命中缓存即跳过）：
  data/ohlcv/target/ohlcv.parquet       列: date, instrument, open, high, low, close, volume, total_turnover, vwap
  data/ohlcv/target/membership.parquet  列: date, instrument, in_csi300, in_csi800
  data/ohlcv/source/ohlcv.parquet       同上（全 A 股，无 membership）
  data/industry/{date}/mapping.parquet            列: 含 first_industry_code（CITIC 一级目录）
  data/industry/{date}/instrument_industry.parquet 列: instrument, first_industry_code
  data/benchmark/000300.XSHG.parquet     列: date, open（前复权、含停牌）
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import ymm_data_sdk

# 目标域指数：中证 800 = 沪深 300 + 中证 500（已验证成分数 300 / 800）
TARGET_INDEXES = {
    "000300.XSHG": "in_csi300",
    "000906.XSHG": "in_csi800",
}
OHLCV_FIELDS = ["open", "high", "low", "close", "volume", "total_turnover"]
DEFAULT_DATA_DIR = Path(__file__).parent / "data"


# ── SDK 初始化 ────────────────────────────────────────────────────────────

def init_sdk() -> None:
    """从 .env 读取 DATA_TOKEN 并初始化 SDK（幂等，mode="lan"）。"""
    if getattr(init_sdk, "_done", False):
        return
    token = _load_token()
    if not token:
        raise RuntimeError("未找到 DATA_TOKEN；请在项目 .env 中配置")
    ymm_data_sdk.init(token=token, mode="lan", warmup=False)
    init_sdk._done = True


def _load_token() -> str:
    candidates = [
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / ".env",  # 项目根 diffusion_transformer/.env
    ]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                if key.strip() == "DATA_TOKEN":
                    return value.strip().strip('"').strip("'")
    return ""


# ── 缓存辅助 ──────────────────────────────────────────────────────────────

def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _read_manifest(folder: Path) -> dict | None:
    path = folder / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(folder: Path, manifest: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(folder / "manifest.json")


# ── 取数 ──────────────────────────────────────────────────────────────────

def fetch_ohlcv(
    instruments: Sequence[str],
    start_date: str,
    end_date: str,
    adjust_type: str = "post",
    skip_suspended: bool = True,
    chunk_size: int = 500,
) -> pd.DataFrame:
    """拉取日频 OHLCV（后复权、剔停牌），派生 vwap = total_turnover / volume。

    返回列: date, instrument, open, high, low, close, volume, total_turnover, vwap
    """
    init_sdk()
    instruments = sorted({str(i) for i in instruments})
    frames: list[pd.DataFrame] = []
    for start in range(0, len(instruments), chunk_size):
        chunk = instruments[start:start + chunk_size]
        df = ymm_data_sdk.get_price(
            chunk, start_date=start_date, end_date=end_date,
            frequency="1d", fields=OHLCV_FIELDS,
            adjust_type=adjust_type, skip_suspended=skip_suspended, expect_df=True,
        )
        if df is not None and not df.empty:
            frames.append(df.reset_index())
    if not frames:
        raise RuntimeError(f"get_price 无数据: {start_date} ~ {end_date}")
    ohlcv = pd.concat(frames, ignore_index=True).rename(columns={"order_book_id": "instrument"})
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    ohlcv["vwap"] = np.divide(
        ohlcv["total_turnover"], ohlcv["volume"],
        out=np.full(len(ohlcv), np.nan, dtype=float),
        where=ohlcv["volume"].to_numpy(dtype=np.float64) > 0,
    )
    cols = ["date", "instrument", "open", "high", "low", "close", "volume", "total_turnover", "vwap"]
    return ohlcv[cols]


def fetch_target_constituents(start_date: str, end_date: str) -> pd.DataFrame:
    """CSI300/CSI800 成分股 point-in-time 成员表。

    返回: MultiIndex (date, instrument) -> in_csi300 / in_csi800 布尔
    """
    init_sdk()
    frames: list[pd.DataFrame] = []
    for index_code, flag in TARGET_INDEXES.items():
        components = ymm_data_sdk.index_components(
            index_code, start_date=start_date, end_date=end_date
        )
        rows = [
            (pd.Timestamp(d), str(i))
            for d, members in components.items()
            for i in members
        ]
        df = pd.DataFrame(rows, columns=["date", "instrument"])
        df[flag] = True
        frames.append(df)
    membership = frames[0]
    for other in frames[1:]:
        membership = membership.merge(other, on=["date", "instrument"], how="outer")
    for flag in TARGET_INDEXES.values():
        membership[flag] = membership[flag].eq(True)
    return membership.set_index(["date", "instrument"]).sort_index()


def fetch_source_universe(start_date: str, end_date: str) -> list[str]:
    """全 A 股股票池（半年度快照并集，近似口径，需在 evidence 中声明）。"""
    init_sdk()
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    snapshots: set[pd.Timestamp] = set()
    for year in range(start.year, end.year + 1):
        for month, day in ((6, 30), (12, 31)):
            d = min(pd.Timestamp(year, month, day), end)
            if d >= start:
                snapshots.add(d)
    instruments: set[str] = set()
    for d in sorted(snapshots):
        frame = ymm_data_sdk.all_instruments(type="CS", date=d.date().isoformat())
        instruments.update(frame["order_book_id"].astype(str))
    return sorted(i for i in instruments if i.endswith((".XSHG", ".XSHE")))


def fetch_limit_status(
    instruments: Sequence[str],
    start_date: str,
    end_date: str,
    chunk_size: int = 500,
) -> pd.DataFrame:
    """未复权空间拉取涨跌停状态（涨跌停是未复权价现象，须 adjust_type="none"）。

    实测口径：adjust_type="none" 下 `limit_up == round(prev_close*1.1, 2)`、`limit_down == round(prev_close*0.9, 2)` 精确成立；
    复权模式下 prev_close 仍为未复权价，limit_up/down 却随复权缩放，二者不可比。

    返回列: date, instrument, is_limit_up, is_limit_down
    """
    init_sdk()
    instruments = sorted({str(i) for i in instruments})
    frames: list[pd.DataFrame] = []
    for start in range(0, len(instruments), chunk_size):
        chunk = instruments[start:start + chunk_size]
        df = ymm_data_sdk.get_price(
            chunk, start_date=start_date, end_date=end_date,
            fields=["close", "limit_up", "limit_down"],
            adjust_type="none", skip_suspended=True, expect_df=True,
        )
        if df is not None and not df.empty:
            frames.append(df.reset_index().rename(columns={"order_book_id": "instrument"}))
    if not frames:
        raise RuntimeError(f"get_price 涨跌停无数据: {start_date} ~ {end_date}")
    limit = pd.concat(frames, ignore_index=True)
    limit["date"] = pd.to_datetime(limit["date"])
    # 涨停/跌停：收盘触及当日涨/跌停价（同口径未复权，round 到分后精确比较）
    limit["is_limit_up"] = (limit["close"].round(2) == limit["limit_up"].round(2)) & (limit["limit_up"] > 0)
    limit["is_limit_down"] = (limit["close"].round(2) == limit["limit_down"].round(2)) & (limit["limit_down"] > 0)
    return limit[["date", "instrument", "is_limit_up", "is_limit_down"]]


def fetch_listed_dates(instruments: Sequence[str]) -> pd.DataFrame:
    """静态上市日期映射: instrument -> listed_date。"""
    init_sdk()
    frame = ymm_data_sdk.all_instruments(type="CS")
    frame = frame[["order_book_id", "listed_date"]].rename(columns={"order_book_id": "instrument"})
    frame["instrument"] = frame["instrument"].astype(str)
    frame = frame[frame["instrument"].isin({str(i) for i in instruments})]
    frame["listed_date"] = pd.to_datetime(frame["listed_date"], errors="coerce")
    return frame


# ── 行业与基准 ────────────────────────────────────────────────────────────

def fetch_industry_mapping(date: str, source: str = "citics_2019") -> pd.DataFrame:
    """CITIC 一级行业目录快照（含 first_industry_code 列的目录表，非股票→行业映射）。"""
    init_sdk()
    return ymm_data_sdk.get_industry_mapping(source=source, date=date, market="cn")


def fetch_instrument_industries(
    instruments: Sequence[str],
    date: str,
    source: str = "citics_2019",
    level: int = 1,
    chunk_size: int = 1000,
) -> pd.Series:
    """股票 → CITIC 一级行业代码。

    返回 Series：index=instrument（order_book_id），values=first_industry_code（str）。
    """
    init_sdk()
    instruments = sorted({str(i) for i in instruments})
    parts: list[pd.Series] = []
    for start in range(0, len(instruments), chunk_size):
        chunk = instruments[start:start + chunk_size]
        frame = ymm_data_sdk.get_instrument_industry(
            chunk, source=source, level=level, date=date, market="cn"
        )
        if frame is None or frame.empty:
            continue
        parts.append(frame["first_industry_code"].astype(str))
    if not parts:
        return pd.Series(dtype=str)
    return pd.concat(parts)


def fetch_benchmark_index(
    start_date: str,
    end_date: str,
    index_code: str = "000300.XSHG",
    adjust_type: str = "pre",
) -> pd.DataFrame:
    """沪深300 指数日频 open（前复权、含停牌），用于基准收益曲线。返回列: date, open。"""
    init_sdk()
    index_bars = ymm_data_sdk.get_price(
        [index_code], start_date=start_date, end_date=end_date,
        frequency="1d", fields=["open"], adjust_type=adjust_type,
        skip_suspended=False, expect_df=True, market="cn",
    ).reset_index()
    index_bars["date"] = pd.to_datetime(index_bars["date"])
    return index_bars.sort_values("date")[["date", "open"]]


def build_industry_data(
    date: str,
    out_dir: str | Path | None = None,
    source: str = "citics_2019",
    force: bool = False,
) -> pd.Series:
    """落盘 data/industry/{date}/（mapping.parquet + instrument_industry.parquet）。

    股票池 = data/ohlcv/source/ohlcv.parquet（全 A 股，剔 ST）∪ data/ohlcv/target/membership.parquet
    （目标域成员），覆盖 diffsformer_data 所需的全部行业 lookup。行业为 CITIC 一级（level=1）。
    """
    out_dir = Path(out_dir) if out_dir else DEFAULT_DATA_DIR
    folder = out_dir / "industry" / date

    instruments: set[str] = set()
    for path in (
        out_dir / "ohlcv" / "source" / "ohlcv.parquet",
        out_dir / "ohlcv" / "target" / "membership.parquet",
    ):
        if path.exists():
            instruments |= set(pd.read_parquet(path, columns=["instrument"])["instrument"].astype(str))
    instruments = sorted(instruments)
    if not instruments:
        raise FileNotFoundError("未找到 data/ohlcv/{source,target}，请先运行 main.py 构建 OHLCV")

    payload = {"kind": "industry", "date": date, "source": source, "n_instruments": len(instruments)}
    fingerprint = _fingerprint(payload)
    manifest = _read_manifest(folder)
    if not force and manifest and manifest.get("fingerprint") == fingerprint and manifest.get("complete"):
        print(f"[market_data] 命中缓存 {folder}")
        return pd.read_parquet(folder / "instrument_industry.parquet")["first_industry_code"]

    mapping = fetch_industry_mapping(date, source=source)
    series = fetch_instrument_industries(instruments, date, source=source)
    folder.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(folder / "mapping.parquet", index=False)
    series.rename("first_industry_code").rename_axis("instrument").reset_index().to_parquet(
        folder / "instrument_industry.parquet", index=False
    )
    _write_manifest(folder, {
        **payload, "fingerprint": fingerprint, "complete": True,
        "n_mapped": int(len(series)),
    })
    print(f"[market_data] 已落盘 {folder}（{len(instruments):,} 只股票 → 行业，映射 {len(series):,} 条）")
    return series


def build_benchmark(
    start_date: str,
    end_date: str,
    out_dir: str | Path | None = None,
    index_code: str = "000300.XSHG",
    adjust_type: str = "pre",
    force: bool = False,
) -> pd.DataFrame:
    """落盘 data/benchmark/{index_code}.parquet（date, open，前复权、含停牌）。"""
    out_dir = Path(out_dir) if out_dir else DEFAULT_DATA_DIR
    folder = out_dir / "benchmark"
    payload = {
        "kind": "benchmark", "index_code": index_code,
        "start_date": start_date, "end_date": end_date, "adjust_type": adjust_type,
    }
    fingerprint = _fingerprint(payload)
    manifest = _read_manifest(folder)
    if not force and manifest and manifest.get("fingerprint") == fingerprint and manifest.get("complete"):
        print(f"[market_data] 命中缓存 {folder}")
        return pd.read_parquet(folder / f"{index_code}.parquet")

    bars = fetch_benchmark_index(start_date, end_date, index_code=index_code, adjust_type=adjust_type)
    folder.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(folder / f"{index_code}.parquet", index=False)
    _write_manifest(folder, {
        **payload, "fingerprint": fingerprint, "complete": True, "n_bars": int(len(bars)),
    })
    print(f"[market_data] 已落盘 {folder / (index_code + '.parquet')}（{len(bars):,} 根）")
    return bars


def _build_limit_status(instruments: Sequence[str], start_date: str, end_date: str) -> pd.DataFrame:
    """涨跌停状态 + 上市日期，合并为 (date, instrument) 表。"""
    limit = fetch_limit_status(instruments, start_date, end_date)
    listed = fetch_listed_dates(instruments)
    return limit.merge(listed, on="instrument", how="left")


# ── 清洗 ──────────────────────────────────────────────────────────────────

def drop_st(ohlcv: pd.DataFrame, chunk_size: int = 1000) -> pd.DataFrame:
    """逐日 point-in-time 剔除 ST / 退市风险警示样本。"""
    init_sdk()
    instruments = sorted(ohlcv["instrument"].astype(str).unique())
    start = pd.Timestamp(ohlcv["date"].min()).date().isoformat()
    end = pd.Timestamp(ohlcv["date"].max()).date().isoformat()
    st_keys: set[tuple[str, str]] = set()
    for i in range(0, len(instruments), chunk_size):
        chunk = instruments[i:i + chunk_size]
        st = ymm_data_sdk.is_st_stock(chunk, start_date=start, end_date=end)
        if st is None or st.empty:
            continue
        for (d, inst), is_st in st.stack().items():
            if bool(is_st):
                st_keys.add((pd.Timestamp(d).strftime("%Y-%m-%d"), str(inst)))
    if not st_keys:
        return ohlcv
    key = pd.MultiIndex.from_arrays(
        [ohlcv["date"].dt.strftime("%Y-%m-%d"), ohlcv["instrument"].astype(str)]
    )
    keep = ~key.isin(pd.MultiIndex.from_tuples(sorted(st_keys)))
    return ohlcv.loc[keep].reset_index(drop=True)


# ── 组装与落盘 ────────────────────────────────────────────────────────────

def build_target_ohlcv(
    start_date: str,
    end_date: str,
    out_dir: str | Path | None = None,
    adjust_type: str = "post",
    force: bool = False,
) -> pd.DataFrame:
    """组装目标域（CSI300∪CSI800）清洗后 OHLCV，落盘 data/ohlcv/target/。

    返回带 in_csi300/in_csi800/is_member 列的 OHLCV（兼容旧 diffsformer_data 路径）。
    """
    out_dir = Path(out_dir) if out_dir else DEFAULT_DATA_DIR
    folder = out_dir / "ohlcv" / "target"
    payload = {
        "kind": "target_ohlcv", "start_date": start_date, "end_date": end_date,
        "adjust_type": adjust_type, "indexes": sorted(TARGET_INDEXES),
    }
    fingerprint = _fingerprint(payload)
    manifest = _read_manifest(folder)
    if not force and manifest and manifest.get("fingerprint") == fingerprint and manifest.get("complete"):
        print(f"[market_data] 命中缓存 {folder}")
        return _attach_membership(
            pd.read_parquet(folder / "ohlcv.parquet"),
            pd.read_parquet(folder / "membership.parquet"),
        )

    membership = fetch_target_constituents(start_date, end_date)
    instruments = sorted(membership.index.get_level_values("instrument").unique())
    print(f"[market_data] 目标域成分股并集: {len(instruments)} 只")
    ohlcv = fetch_ohlcv(instruments, start_date, end_date, adjust_type=adjust_type)
    n0 = len(ohlcv)
    ohlcv = drop_st(ohlcv)
    print(f"[market_data] 剔除 ST 样本: {n0 - len(ohlcv):,} / {n0:,}")

    limit_status = _build_limit_status(instruments, start_date, end_date)
    folder.mkdir(parents=True, exist_ok=True)
    ohlcv.to_parquet(folder / "ohlcv.parquet", index=False)
    membership.reset_index().to_parquet(folder / "membership.parquet", index=False)
    limit_status.to_parquet(folder / "limit_status.parquet", index=False)
    _write_manifest(folder, {
        **payload, "fingerprint": fingerprint, "complete": True,
        "n_instruments": len(instruments), "n_rows": int(len(ohlcv)),
        "n_st_dropped": int(n0 - len(ohlcv)),
    })
    print(f"[market_data] 已落盘 {folder}（{len(ohlcv):,} 行）")
    return _attach_membership(ohlcv, membership)


def _attach_membership(ohlcv: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    members = membership.reset_index()
    out = ohlcv.merge(members, on=["date", "instrument"], how="left")
    for flag in TARGET_INDEXES.values():
        out[flag] = out[flag].eq(True)
    out["is_member"] = out["in_csi300"] | out["in_csi800"]
    return out


def build_source_ohlcv(
    start_date: str,
    end_date: str,
    out_dir: str | Path | None = None,
    adjust_type: str = "post",
    force: bool = False,
) -> pd.DataFrame:
    """组装源域（全 A 股）清洗后 OHLCV，落盘 data/ohlcv/source/。"""
    out_dir = Path(out_dir) if out_dir else DEFAULT_DATA_DIR
    folder = out_dir / "ohlcv" / "source"
    payload = {
        "kind": "source_ohlcv", "start_date": start_date, "end_date": end_date,
        "adjust_type": adjust_type,
    }
    fingerprint = _fingerprint(payload)
    manifest = _read_manifest(folder)
    if not force and manifest and manifest.get("fingerprint") == fingerprint and manifest.get("complete"):
        print(f"[market_data] 命中缓存 {folder}")
        return pd.read_parquet(folder / "ohlcv.parquet")

    instruments = fetch_source_universe(start_date, end_date)
    print(f"[market_data] 源域全 A 股: {len(instruments)} 只")
    ohlcv = fetch_ohlcv(instruments, start_date, end_date, adjust_type=adjust_type)
    n0 = len(ohlcv)
    ohlcv = drop_st(ohlcv)
    print(f"[market_data] 剔除 ST 样本: {n0 - len(ohlcv):,} / {n0:,}")

    limit_status = _build_limit_status(instruments, start_date, end_date)
    folder.mkdir(parents=True, exist_ok=True)
    ohlcv.to_parquet(folder / "ohlcv.parquet", index=False)
    limit_status.to_parquet(folder / "limit_status.parquet", index=False)
    _write_manifest(folder, {
        **payload, "fingerprint": fingerprint, "complete": True,
        "n_instruments": len(instruments), "n_rows": int(len(ohlcv)),
        "n_st_dropped": int(n0 - len(ohlcv)),
    })
    print(f"[market_data] 已落盘 {folder}（{len(ohlcv):,} 行）")
    return ohlcv
