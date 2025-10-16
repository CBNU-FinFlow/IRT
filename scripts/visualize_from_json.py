#!/usr/bin/env python3
"""FinFlow XAI 시각화 리포트 생성 스크립트."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Matplotlib 캐시 경고 방지용 임시 디렉토리 설정
_runtime_cache = Path.cwd() / ".finflow_cache"
(_runtime_cache / "mpl").mkdir(parents=True, exist_ok=True)
(_runtime_cache / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str((_runtime_cache / "mpl").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((_runtime_cache / "xdg").resolve()))

import matplotlib.pyplot as plt

from finrl.evaluation.visualizer import (
    compute_drawdown,
    compute_equity_curve,
    compute_rolling_stats,
    ensure_datetime_index,
    normalise_attributions,
    resample_matrix,
    plot_attr_heatmap,
    plot_attr_regime_bars,
    plot_cash_vs_crisis,
    plot_cost_footprint,
    plot_crisis_kappa_cash,
    plot_drawdown,
    plot_entropy_temp,
    plot_equity,
    plot_proto_area,
    plot_returns_dist,
    plot_rolling_risk_return,
    plot_turnover_vs_reward,
)


def _is_non_empty(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.size > 0
    if isinstance(value, (np.ndarray, np.generic)):
        return np.asarray(value).size > 0
    return True


def _first_non_empty(*values):
    for value in values:
        if _is_non_empty(value):
            return value
    return None


@dataclass
class PlotConfig:
    include_core: bool
    include_xai: bool
    crisis_threshold: Optional[float]
    crisis_quantile: Optional[float]
    max_topk: int
    heatmap_steps: int
    benchmark_mode: str
    img_format: str
    dpi: int
    rolling_window: int
    html_index: bool


@dataclass
class ReportData:
    returns: pd.Series
    equity: pd.Series
    drawdown: pd.Series
    benchmark: Optional[pd.Series]
    metrics: Dict[str, Optional[float]]
    rolling: Optional[object]
    crisis: Optional[pd.Series]
    cash_ratio: Optional[pd.Series]
    kappa_sharpe: Optional[pd.Series]
    kappa_cvar: Optional[pd.Series]
    turnover: Optional[pd.Series]
    rewards: pd.Series
    tx_cost: Optional[pd.Series]
    proto_weights: Optional[pd.DataFrame]
    proto_entropy: Optional[pd.Series]
    action_temp: Optional[pd.Series]
    alpha_c: Optional[pd.Series]
    attr_matrix: Optional[pd.DataFrame]
    attr_note: str
    regime_marks: Optional[pd.Series]
    topk_normal: Dict[str, float]
    topk_crisis: Dict[str, float]
    crisis_threshold: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FinFlow 평가 산출물을 기반으로 정적 XAI 리포트를 생성합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True, help="평가 산출물이 위치한 디렉토리"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="시각화 출력 디렉토리 (기본: <input-dir>/viz)",
    )
    parser.add_argument(
        "--include-core", dest="include_core", action="store_true", default=True
    )
    parser.add_argument("--no-core", dest="include_core", action="store_false")
    parser.add_argument(
        "--include-xai", dest="include_xai", action="store_true", default=True
    )
    parser.add_argument("--no-xai", dest="include_xai", action="store_false")
    parser.add_argument("--crisis-threshold", type=float, default=0.5)
    parser.add_argument("--crisis-quantile", type=float, default=None)
    parser.add_argument("--max-topk", type=int, default=5)
    parser.add_argument("--heatmap-steps", type=int, default=300)
    parser.add_argument("--benchmark", choices=["auto", "none"], default="auto")
    parser.add_argument(
        "--format", dest="img_format", choices=["png", "pdf"], default="png"
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--rolling-window", type=int, default=60)
    parser.add_argument(
        "--html-index", action="store_true", help="HTML 썸네일 인덱스 생성"
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    if path.is_file():
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    return {}


def _resolve_path(
    root: Path, candidate: Optional[str], *fallbacks: str
) -> Optional[Path]:
    candidates: List[Path] = []
    if candidate:
        c_path = Path(candidate)
        candidates.append(c_path if c_path.is_absolute() else root / c_path)
    candidates.extend(root / fb for fb in fallbacks)
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_prototypes(path: Optional[Path]) -> Optional[pd.DataFrame]:
    df = _load_csv(path)
    if df is not None and not df.empty:
        return df
    return None


def _load_attributions(
    parquet_path: Optional[Path], csv_path: Optional[Path]
) -> Optional[pd.DataFrame]:
    if parquet_path is not None and parquet_path.is_file():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_path is not None and csv_path.is_file():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass
    return None


def _map_index_to_dates(
    series: pd.Series, mapper: Dict[int, pd.Timestamp]
) -> pd.Series:
    if series.empty:
        return series
    mapped = [mapper.get(int(idx), idx) for idx in series.index]
    series.index = pd.Index(mapped)
    return series.sort_index()


def _map_df_index_to_dates(
    df: pd.DataFrame, mapper: Dict[int, pd.Timestamp]
) -> pd.DataFrame:
    if df.empty:
        return df
    mapped = [mapper.get(int(idx), idx) for idx in df.index]
    df.index = pd.Index(mapped)
    return df.sort_index()


def _extract_topk(summary: Dict, regime: str, limit: int) -> Dict[str, float]:
    section = summary.get(regime) if summary else None
    if not section:
        return {}
    candidates = section.get("top_features") or section.get("topk_features")
    if not candidates:
        return {}
    result: Dict[str, float] = {}
    if isinstance(candidates, dict):
        items = list(candidates.items())
    else:
        items = []
        for entry in candidates:
            if isinstance(entry, dict):
                name = entry.get("feature") or entry.get("name")
                value = (
                    entry.get("score") or entry.get("importance") or entry.get("weight")
                )
                if name is not None and value is not None:
                    items.append((str(name), float(value)))
    for name, value in items:
        if name not in result:
            result[name] = float(value)
        if len(result) >= limit:
            break
    return result


def _prepare_proto_weights(
    df: Optional[pd.DataFrame], cfg: PlotConfig, date_map: Dict[int, pd.Timestamp]
) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    step_col = next(
        (c for c in df.columns if c.lower() in {"step", "t", "index"}), None
    )
    if step_col is None:
        return None
    records: List[pd.Series] = []
    for _, row in df.iterrows():
        step_val = int(row[step_col])
        weights: Dict[str, float] = {}
        other_total = 0.0
        for rank in range(1, cfg.max_topk + 1):
            w_col = f"topk_{rank}_weight"
            if w_col not in row:
                continue
            weight = float(row[w_col]) if np.isfinite(row[w_col]) else 0.0
            if weight <= 0:
                continue
            id_col = f"topk_{rank}_id"
            label = (
                str(row[id_col])
                if id_col in row and pd.notna(row[id_col])
                else f"Proto {rank}"
            )
            weights[label] = weight
            other_total += weight
        if weights:
            weights["Other"] = max(0.0, 1.0 - other_total)
            records.append(pd.Series(weights, name=step_val))
    if not records:
        return None
    proto_df = pd.DataFrame(records).fillna(0.0)
    proto_df = proto_df.groupby(proto_df.index).mean()
    mapped_index = [date_map.get(int(idx), idx) for idx in proto_df.index]
    proto_df.index = pd.Index(mapped_index)
    return proto_df.sort_index()


def build_report_data(root: Path, cfg: PlotConfig) -> Optional[ReportData]:
    evaluation = _load_json(root / "evaluation_results.json")
    insights = _load_json(root / "evaluation_insights.json")
    payload = evaluation.get("results", evaluation)
    series = payload.get("series", {})
    metrics_full = payload.get("metrics", {})

    per_step_returns = _first_non_empty(
        series.get("per_step_returns"),
        series.get("returns"),
        insights.get("per_step_returns"),
    )
    if not _is_non_empty(per_step_returns):
        print("[viz] Error: 평가 결과에서 per_step_returns를 찾을 수 없습니다.")
        return None

    returns = np.asarray(per_step_returns, dtype=float)
    steps = np.arange(returns.size)
    date_candidates = _first_non_empty(series.get("dates"), insights.get("dates"))
    if _is_non_empty(date_candidates):
        date_index = ensure_datetime_index(date_candidates)
    else:
        date_index = pd.Index(steps, name="step")
    step_to_date = {int(step): date for step, date in zip(steps, date_index)}

    returns_series = pd.Series(returns, index=date_index, name="returns")
    equity = compute_equity_curve(returns, index=date_index)
    drawdown = compute_drawdown(equity)
    rolling = None
    if returns.size >= max(5, cfg.rolling_window):
        rolling = compute_rolling_stats(
            returns, index=date_index, window=cfg.rolling_window
        )

    benchmark = None
    if cfg.benchmark_mode == "auto":
        benchmark_candidates = _first_non_empty(
            series.get("benchmark_returns"), insights.get("benchmark_returns")
        )
        if _is_non_empty(benchmark_candidates):
            bench_arr = np.asarray(benchmark_candidates, dtype=float)
            length = min(len(bench_arr), len(date_index))
            benchmark = compute_equity_curve(
                bench_arr[:length], index=date_index[:length]
            )

    metrics_summary = {
        "Sharpe": metrics_full.get("sharpe_ratio"),
        "Sortino": metrics_full.get("sortino_ratio"),
        "Annual return": metrics_full.get("annualized_return"),
        "Max drawdown": metrics_full.get("max_drawdown"),
        "CVaR@5%": metrics_full.get("cvar_5"),
        "Total return": metrics_full.get("total_return"),
    }

    crisis_series = None
    crisis_candidates = _first_non_empty(
        series.get("crisis_levels"), payload.get("irt", {}).get("crisis_levels")
    )
    if _is_non_empty(crisis_candidates):
        crisis_array = np.asarray(crisis_candidates, dtype=float)
        crisis_series = pd.Series(crisis_array, index=date_index[: len(crisis_array)])
        crisis_series = crisis_series.reindex(date_index, method="pad")

    kappa_sharpe = None
    irt_payload = payload.get("irt", {}) or {}
    kappa_candidates = _first_non_empty(
        irt_payload.get("env_kappa"), series.get("kappa_sharpe")
    )
    if _is_non_empty(kappa_candidates):
        arr = np.asarray(kappa_candidates, dtype=float)
        kappa_sharpe = pd.Series(arr, index=date_index[: len(arr)]).reindex(
            date_index, method="pad"
        )

    kappa_cvar = None
    kappa_cvar_candidates = series.get("kappa_cvar")
    if _is_non_empty(kappa_cvar_candidates):
        arr = np.asarray(kappa_cvar_candidates, dtype=float)
        kappa_cvar = pd.Series(arr, index=date_index[: len(arr)]).reindex(
            date_index, method="pad"
        )

    paths = payload.get("paths", {}) or {}
    holdings_path = _resolve_path(
        root, paths.get("holdings_csv"), "holdings_timeseries.csv"
    )
    trades_path = _resolve_path(root, paths.get("trades_csv"), "trades.csv")
    prototypes_path = _resolve_path(root, None, "xai/xai_prototypes_timeseries.csv")
    attr_parquet_path = _resolve_path(
        root, None, "xai/xai_feature_attributions.parquet"
    )
    attr_csv_path = _resolve_path(root, None, "xai/xai_feature_attributions.csv")
    summary_path = _resolve_path(root, None, "xai/xai_summary.json")

    holdings_df = _load_csv(holdings_path)
    cash_ratio = None
    if _is_non_empty(series.get("cash_ratio_series")):
        arr = np.asarray(series["cash_ratio_series"], dtype=float)
        cash_ratio = pd.Series(arr, index=date_index[: len(arr)]).reindex(
            date_index, method="pad"
        )
    elif (
        holdings_df is not None
        and not holdings_df.empty
        and "CASH" in holdings_df.columns
    ):
        if "step" in holdings_df.columns:
            grouped = holdings_df.groupby("step")["CASH"].mean()
            cash_ratio = pd.Series(
                grouped.values,
                index=[step_to_date.get(int(idx), idx) for idx in grouped.index],
            ).reindex(date_index, method="pad")
        elif "date" in holdings_df.columns:
            cash_ratio = pd.Series(
                holdings_df["CASH"].astype(float).values,
                index=ensure_datetime_index(holdings_df["date"]),
            ).reindex(date_index, method="pad")

    trades_df = _load_csv(trades_path)
    turnover = None
    tx_cost = None
    if _is_non_empty(series.get("turnover_executed")):
        arr = np.asarray(series["turnover_executed"], dtype=float)
        turnover = pd.Series(arr, index=date_index[: len(arr)]).reindex(
            date_index, method="pad"
        )
    elif trades_df is not None and not trades_df.empty:
        step_col = next(
            (c for c in trades_df.columns if c.lower() in {"step", "episode_step"}),
            None,
        )
        qty_col = next(
            (c for c in trades_df.columns if c.lower() in {"qty", "quantity"}), None
        )
        price_col = next((c for c in trades_df.columns if c.lower() == "price"), None)
        if step_col and qty_col and price_col:
            trades_df[step_col] = pd.to_numeric(trades_df[step_col], errors="coerce")
            trades_df = trades_df.dropna(subset=[step_col])
            grouped = (
                (trades_df[qty_col].abs() * trades_df[price_col].abs())
                .groupby(trades_df[step_col])
                .sum()
            )
            turnover = _map_index_to_dates(
                pd.Series(grouped.values, index=grouped.index.astype(int)), step_to_date
            ).reindex(date_index, method="pad")
        cost_col = next(
            (
                c
                for c in trades_df.columns
                if c.lower() in {"tx_cost", "transaction_cost", "cost"}
            ),
            None,
        )
        if step_col and cost_col:
            grouped_cost = trades_df.groupby(trades_df[step_col])[cost_col].sum()
            tx_cost = _map_index_to_dates(
                pd.Series(grouped_cost.values, index=grouped_cost.index.astype(int)),
                step_to_date,
            ).reindex(date_index, method="pad")

    if _is_non_empty(series.get("transaction_costs")):
        arr = np.asarray(series["transaction_costs"], dtype=float)
        tx_cost = pd.Series(arr, index=date_index[: len(arr)]).reindex(
            date_index, method="pad"
        )

    proto_df = _prepare_proto_weights(
        _load_prototypes(prototypes_path), cfg, step_to_date
    )
    proto_entropy = None
    action_temp = None
    alpha_c = None
    if _is_non_empty(irt_payload.get("proto_entropy")):
        arr = np.asarray(irt_payload["proto_entropy"], dtype=float)
        proto_entropy = pd.Series(arr, index=date_index[: len(arr)]).reindex(
            date_index, method="pad"
        )
    if _is_non_empty(irt_payload.get("alpha_c")):
        arr = np.asarray(irt_payload["alpha_c"], dtype=float)
        alpha_c = pd.Series(arr, index=date_index[: len(arr)]).reindex(
            date_index, method="pad"
        )
    if (
        proto_df is None
        and proto_entropy is None
        and _is_non_empty(series.get("prototype_weights"))
    ):
        proto_df = pd.DataFrame(series["prototype_weights"])
        proto_df.index = [step_to_date.get(int(idx), idx) for idx in proto_df.index]

    if action_temp is None and _is_non_empty(series.get("action_temp")):
        arr = np.asarray(series["action_temp"], dtype=float)
        action_temp = pd.Series(arr, index=date_index[: len(arr)]).reindex(
            date_index, method="pad"
        )

    attr_raw = _load_attributions(attr_parquet_path, attr_csv_path)
    attr_matrix = None
    attr_note = "Rows normalised so that Σ|attr|=1."
    if attr_raw is not None and not attr_raw.empty:
        if {"step", "feature", "attribution"}.issubset(attr_raw.columns):
            pivot = attr_raw.pivot_table(
                index="step", columns="feature", values="attribution", aggfunc="mean"
            ).fillna(0.0)
            pivot = normalise_attributions(pivot, method="l1")
            pivot = resample_matrix(pivot, cfg.heatmap_steps)
            attr_matrix = _map_df_index_to_dates(pivot, step_to_date)
            attr_matrix = attr_matrix.loc[:, sorted(attr_matrix.columns)]
        else:
            attr_matrix = None

    summary = _load_json(summary_path) if summary_path else {}
    topk_normal = _extract_topk(summary, "normal", cfg.max_topk)
    topk_crisis = _extract_topk(summary, "crisis", cfg.max_topk)

    crisis_threshold = cfg.crisis_threshold
    if (
        cfg.crisis_quantile is not None
        and crisis_series is not None
        and not crisis_series.empty
    ):
        crisis_threshold = float(crisis_series.quantile(cfg.crisis_quantile))

    regime_marks = None
    if crisis_series is not None and not crisis_series.empty:
        regime_marks = (crisis_series >= (crisis_threshold or 0.5)).astype(float)

    if proto_df is not None:
        proto_df = proto_df.reindex(date_index, method="pad")
    if proto_entropy is not None:
        proto_entropy = proto_entropy.reindex(date_index, method="pad")
    if action_temp is not None:
        action_temp = action_temp.reindex(date_index, method="pad")
    if alpha_c is not None:
        alpha_c = alpha_c.reindex(date_index, method="pad")
    if cash_ratio is not None:
        cash_ratio = cash_ratio.reindex(date_index, method="pad")
    if turnover is not None:
        turnover = turnover.reindex(date_index, method="pad")
    if tx_cost is not None:
        tx_cost = tx_cost.reindex(date_index, method="pad")
    if kappa_sharpe is not None:
        kappa_sharpe = kappa_sharpe.reindex(date_index, method="pad")
    if kappa_cvar is not None:
        kappa_cvar = kappa_cvar.reindex(date_index, method="pad")
    if attr_matrix is not None and not attr_matrix.empty:
        attr_matrix = attr_matrix.loc[:, sorted(attr_matrix.columns)]

    rewards_series = returns_series

    return ReportData(
        returns=returns_series,
        equity=equity,
        drawdown=drawdown,
        benchmark=benchmark,
        metrics=metrics_summary,
        rolling=rolling,
        crisis=crisis_series,
        cash_ratio=cash_ratio,
        kappa_sharpe=kappa_sharpe,
        kappa_cvar=kappa_cvar,
        turnover=turnover,
        rewards=rewards_series,
        tx_cost=tx_cost,
        proto_weights=proto_df,
        proto_entropy=proto_entropy,
        action_temp=action_temp,
        alpha_c=alpha_c,
        attr_matrix=attr_matrix,
        attr_note=(
            "Rows normalised so that Σ|attr|=1." if attr_matrix is not None else ""
        ),
        regime_marks=regime_marks,
        topk_normal=topk_normal,
        topk_crisis=topk_crisis,
        crisis_threshold=crisis_threshold,
    )


def _save_figure(fig, path: Path, cfg: PlotConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format=cfg.img_format, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)


def generate_plots(data: ReportData, cfg: PlotConfig, out_dir: Path) -> List[Path]:
    outputs: List[Path] = []

    def emit(slug: str, fig) -> None:
        if fig is None:
            return
        filename = f"{len(outputs)+1:02d}_{slug}.{cfg.img_format}"
        path = out_dir / filename
        _save_figure(fig, path, cfg)
        outputs.append(path)

    if cfg.include_core:
        emit(
            "equity_vs_benchmark",
            plot_equity(
                data.equity, benchmark=data.benchmark, annotations=data.metrics
            ),
        )
        emit("drawdown_curve", plot_drawdown(data.drawdown))
        emit(
            "returns_distribution",
            plot_returns_dist(data.returns, cvar_5=data.metrics.get("CVaR@5%")),
        )
        if data.rolling is not None:
            emit(
                "rolling_risk_return",
                plot_rolling_risk_return(
                    data.rolling.sharpe,
                    data.rolling.volatility,
                    cfg.rolling_window,
                ),
            )

    if cfg.include_xai:
        if data.crisis is not None and data.cash_ratio is not None:
            emit(
                "crisis_cash_overlay",
                plot_crisis_kappa_cash(
                    data.crisis,
                    data.cash_ratio,
                    kappa_sharpe=data.kappa_sharpe,
                    kappa_cvar=data.kappa_cvar,
                    threshold=data.crisis_threshold,
                ),
            )
            emit(
                "cash_vs_crisis_density",
                plot_cash_vs_crisis(data.crisis, data.cash_ratio),
            )
        if data.turnover is not None:
            emit(
                "turnover_vs_reward",
                plot_turnover_vs_reward(data.turnover, data.rewards),
            )
        if data.tx_cost is not None and not data.tx_cost.empty:
            emit("transaction_cost_footprint", plot_cost_footprint(data.tx_cost))
        if data.proto_weights is not None and not data.proto_weights.empty:
            emit("prototype_weights_topk", plot_proto_area(data.proto_weights))
        if data.proto_entropy is not None:
            emit(
                "entropy_temperature",
                plot_entropy_temp(
                    data.proto_entropy,
                    action_temp=data.action_temp,
                    alpha_c=data.alpha_c,
                ),
            )
        if data.attr_matrix is not None and not data.attr_matrix.empty:
            emit(
                "feature_attr_heatmap",
                plot_attr_heatmap(
                    data.attr_matrix,
                    feature_order=list(data.attr_matrix.columns),
                    regime_marks=data.regime_marks,
                    normalisation_note=data.attr_note,
                ),
            )
        if data.topk_normal or data.topk_crisis:
            emit(
                "feature_attr_regimes",
                plot_attr_regime_bars(data.topk_normal, data.topk_crisis),
            )

    return outputs


def write_html_index(out_dir: Path, image_paths: List[Path]) -> None:
    if not image_paths:
        return
    lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>FinFlow XAI Report</title>",
        "<style>body{font-family:Arial, sans-serif;margin:20px;}figure{margin-bottom:30px;}img{max-width:100%;height:auto;border:1px solid #ccc;}figcaption{margin-top:8px;color:#555;}</style>",
        "</head>",
        "<body>",
        "<h1>FinFlow XAI Visualization</h1>",
    ]
    for path in image_paths:
        slug = path.name
        lines.extend(
            [
                "<figure>",
                f"<img src='{slug}' alt='{slug}'>",
                f"<figcaption>{slug}</figcaption>",
                "</figure>",
            ]
        )
    lines.extend(["</body>", "</html>"])
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")

    output_dir = (
        Path(args.out_dir).expanduser().resolve() if args.out_dir else input_dir / "viz"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = PlotConfig(
        include_core=args.include_core,
        include_xai=args.include_xai,
        crisis_threshold=args.crisis_threshold,
        crisis_quantile=args.crisis_quantile,
        max_topk=max(1, args.max_topk),
        heatmap_steps=max(10, args.heatmap_steps),
        benchmark_mode=args.benchmark,
        img_format=args.img_format,
        dpi=args.dpi,
        rolling_window=max(5, args.rolling_window),
        html_index=args.html_index,
    )

    report_data = build_report_data(input_dir, cfg)
    if report_data is None:
        print("[viz] 데이터 구성이 실패했습니다. 산출물이 충분한지 확인하세요.")
        return

    generated = generate_plots(report_data, cfg, output_dir)
    if not generated:
        print("[viz] 생성된 플롯이 없습니다.")
    else:
        print(f"[viz] {len(generated)}개 플롯 생성: {output_dir}")
    if cfg.html_index and generated:
        write_html_index(output_dir, generated)
        print(f"[viz] index.html 생성 완료: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
