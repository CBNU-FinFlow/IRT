"""
FinFlow XAI visualization utilities.

This module provides reusable plotting functions dedicated to the
evaluation-time explainability report.  The design principle is to
surface cause-and-effect signals: how risk sensors, portfolio actions,
and learned representations interact with performance outcomes.

Each plot function returns a Matplotlib ``Figure`` so that callers can
decide where and how to persist the artefact.  Helper utilities are
included for common data preparation tasks (equity curve, drawdown,
rolling metrics, attribution normalisation, …).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _to_series(
    values: Optional[Sequence[float]],
    index: Optional[Iterable] = None,
    name: Optional[str] = None,
) -> Optional[pd.Series]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    if index is None:
        index = np.arange(arr.size)
    series = pd.Series(arr, index=pd.Index(index), name=name)
    return series


def _is_non_empty(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (np.ndarray, np.generic)):
        return np.asarray(value).size > 0
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.size > 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _first_non_empty(*values):
    for value in values:
        if _is_non_empty(value):
            return value
    return None


def sanitize_returns(
    returns: Sequence[float],
    cap: float = 0.3,
    floor: float = -0.99,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Clamp and clean return series to prevent runaway spikes while keeping the
    per-step semantics unchanged.
    """
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=fill_value, posinf=cap, neginf=floor)
    if cap is not None:
        arr = np.minimum(arr, cap)
    if floor is not None:
        arr = np.maximum(arr, floor)
    return arr


def compute_equity_curve(
    returns: Sequence[float], base_value: float = 1.0, index: Optional[Iterable] = None
) -> pd.Series:
    returns_series = _to_series(returns, index=index, name="returns")
    if returns_series is None:
        raise ValueError("No returns available for equity curve computation.")
    equity = (1.0 + returns_series).cumprod() * float(base_value)
    equity.name = "equity"
    return equity


def compute_drawdown(equity: pd.Series) -> pd.Series:
    if equity.empty:
        raise ValueError("Equity series is empty.")
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    drawdown.name = "drawdown"
    return drawdown


@dataclass
class RollingStats:
    sharpe: pd.Series
    volatility: pd.Series


def compute_rolling_stats(
    returns: Sequence[float],
    index: Optional[Iterable] = None,
    window: int = 60,
    eps: float = 1e-9,
) -> RollingStats:
    returns_series = _to_series(returns, index=index, name="returns")
    if returns_series is None:
        raise ValueError("Rolling stats require non-empty returns.")
    wins = max(int(window), 2)
    roll_mean = returns_series.rolling(wins).mean()
    roll_std = returns_series.rolling(wins).std(ddof=0).clip(lower=eps)

    sharpe = (roll_mean / roll_std) * np.sqrt(wins)
    sharpe.name = "rolling_sharpe"
    volatility = roll_std * np.sqrt(wins)
    volatility.name = "rolling_volatility"
    return RollingStats(sharpe=sharpe, volatility=volatility)


def normalise_attributions(
    attribution_df: pd.DataFrame,
    method: str = "l1",
) -> pd.DataFrame:
    df = attribution_df.copy()
    if df.empty:
        return df
    if method == "l1":
        denom = df.abs().sum(axis=1).replace(0.0, np.nan)
        df = df.divide(denom, axis=0)
    elif method == "z":
        df = (df - df.mean(axis=1).values[:, None]) / df.std(axis=1).replace(0.0, np.nan)
    else:
        raise ValueError(f"Unknown normalisation method: {method}")
    return df.clip(lower=-1.0, upper=1.0)


def resample_matrix(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if df.empty or len(df) <= max_rows:
        return df
    positions = np.linspace(0, len(df) - 1, num=max_rows, dtype=int)
    sampled = df.iloc[positions]
    return sampled


def ensure_datetime_index(index: Sequence) -> pd.Index:
    try:
        datetimes = pd.to_datetime(index)
        return pd.Index(datetimes)
    except Exception:
        return pd.Index(index)


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

FIGSIZE_DEFAULT = (10, 6)
CAPTION_Y = 0.01


def _new_figure(nrows: int = 1, ncols: int = 1, sharex: bool = False) -> Tuple[Figure, np.ndarray]:
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        figsize=FIGSIZE_DEFAULT,
        constrained_layout=True,
    )
    return fig, axes


def _format_percent_axis(ax: Axes, axis: str = "y") -> None:
    formatter = mticker.PercentFormatter(xmax=1.0, decimals=1)
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def _format_time_axis(ax: Axes, index: pd.Index) -> None:
    if isinstance(index, pd.DatetimeIndex):
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", rotation=0)


def _add_caption(fig: Figure, text: str) -> None:
    fig.text(
        0.01,
        CAPTION_Y,
        text,
        fontsize=9,
        color="#444444",
        ha="left",
        va="bottom",
    )


def _textify_metrics(metrics: Dict[str, Optional[float]]) -> str:
    parts = []
    for key, value in metrics.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        if "drawdown" in key.lower():
            parts.append(f"{key}: {value:+.1%}")
        elif "sharpe" in key.lower() or "sortino" in key.lower():
            parts.append(f"{key}: {value:.2f}")
        elif "return" in key.lower() or "reward" in key.lower():
            parts.append(f"{key}: {value:+.2%}")
        elif "cvar" in key.lower():
            parts.append(f"{key}: {value:+.1%}")
        else:
            parts.append(f"{key}: {value:.3f}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_equity(
    equity: pd.Series,
    benchmark: Optional[pd.Series] = None,
    annotations: Optional[Dict[str, Optional[float]]] = None,
    caption: str = "",
) -> Figure:
    fig, ax = _new_figure()
    idx = equity.index
    ax.plot(idx, equity, label="Portfolio", linewidth=2.0, color="#2b6cb0")
    if benchmark is not None and not benchmark.empty:
        benchmark_aligned = benchmark.reindex(idx, method="pad").dropna()
        ax.plot(
            benchmark_aligned.index,
            benchmark_aligned,
            label="Benchmark",
            linewidth=1.5,
            color="#718096",
            linestyle="--",
        )
    _format_time_axis(ax, idx)
    ax.set_ylabel("Equity (rel.)")
    ax.set_xlabel("Step")
    ax.grid(alpha=0.2)
    if annotations:
        txt = _textify_metrics(annotations)
        if txt:
            ax.text(0.02, 0.95, txt, transform=ax.transAxes, fontsize=10, va="top")
    ax.legend(loc="upper left")
    _add_caption(
        fig,
        caption
        or "Equity curve derived from per_step_returns (portfolio) and benchmark_returns (if available).",
    )
    return fig


def plot_drawdown(drawdown: pd.Series, caption: str = "") -> Figure:
    fig, ax = _new_figure()
    idx = drawdown.index
    ax.fill_between(idx, drawdown, 0.0, color="#c53030", alpha=0.3)
    ax.plot(idx, drawdown, color="#822727", linewidth=1.5)
    _format_time_axis(ax, idx)
    _format_percent_axis(ax, "y")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Step")
    ax.set_title("Drawdown curve")
    ax.grid(alpha=0.2)
    _add_caption(
        fig,
        caption
        or "Drawdown computed as equity / rolling_max - 1.0 using cumulative per_step_returns.",
    )
    return fig


def plot_returns_dist(
    returns: pd.Series,
    cvar_5: Optional[float] = None,
    caption: str = "",
) -> Figure:
    fig, axes = _new_figure(nrows=1, ncols=2, sharex=False)
    axes = np.atleast_1d(axes)
    idx = returns.index
    axes[0].plot(idx, returns * 100, color="#2b6cb0", linewidth=1.2)
    axes[0].axhline(0.0, color="#a0aec0", linewidth=0.8)
    axes[0].set_ylabel("Return (%)")
    axes[0].set_xlabel("Step")
    axes[0].set_title("Per-step returns")
    axes[0].grid(alpha=0.2)
    _format_time_axis(axes[0], idx)

    axes[1].hist(returns * 100, bins=40, color="#63b3ed", alpha=0.8, edgecolor="#2c5282")
    axes[1].set_xlabel("Return (%)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution with CVaR marker")
    if cvar_5 is not None:
        axes[1].axvline(cvar_5 * 100, color="#c53030", linestyle="--", linewidth=2.0)
        axes[1].text(
            cvar_5 * 100,
            axes[1].get_ylim()[1] * 0.9,
            f"CVaR@5% = {cvar_5:.2%}",
            rotation=90,
            color="#c53030",
            va="top",
            ha="right",
        )
    _add_caption(
        fig,
        caption
        or "Returns computed from per_step_returns; CVaR line sourced from evaluation_insights.json (cvar_5).",
    )
    return fig


def plot_rolling_risk_return(
    sharpe: pd.Series,
    volatility: pd.Series,
    window: int,
    caption: str = "",
) -> Figure:
    fig, axes = _new_figure(nrows=2, ncols=1, sharex=True)
    idx = sharpe.index
    axes[0].plot(idx, sharpe, color="#2f855a", linewidth=1.5)
    axes[0].axhline(0.0, color="#a0aec0", linewidth=0.8)
    axes[0].set_ylabel("Rolling Sharpe")
    axes[0].set_title(f"{window}-step rolling Sharpe")
    axes[0].grid(alpha=0.2)

    axes[1].plot(idx, volatility, color="#dd6b20", linewidth=1.5)
    axes[1].set_ylabel("Annualised Volatility")
    axes[1].set_xlabel("Step")
    axes[1].set_title(f"{window}-step rolling volatility (annualised)")
    axes[1].grid(alpha=0.2)
    _format_time_axis(axes[1], idx)
    _add_caption(
        fig,
        caption
        or f"Rolling statistics derived from per_step_returns with window={window}; "
        "Sharpe = mean/std * sqrt(window).",
    )
    return fig


def plot_crisis_kappa_cash(
    crisis: pd.Series,
    cash_ratio: pd.Series,
    kappa_sharpe: Optional[pd.Series] = None,
    kappa_cvar: Optional[pd.Series] = None,
    threshold: Optional[float] = None,
    caption: str = "",
) -> Figure:
    fig, ax = _new_figure()
    idx = crisis.index
    ax.plot(idx, crisis, color="#c53030", linewidth=1.5, label="Crisis level")
    ax.plot(idx, cash_ratio, color="#2b6cb0", linewidth=1.5, label="Cash ratio")
    if threshold is not None:
        ax.axhline(threshold, color="#feb24c", linestyle="--", linewidth=1.0, label="Crisis threshold")
    ax.set_ylabel("Probability / Ratio")
    ax.grid(alpha=0.2)
    _format_time_axis(ax, idx)
    ax.set_xlabel("Step")

    if kappa_sharpe is not None or kappa_cvar is not None:
        ax2 = ax.twinx()
        if kappa_sharpe is not None:
            ax2.plot(idx, kappa_sharpe.reindex(idx), color="#38a169", linewidth=1.2, label="κ Sharpe")
        if kappa_cvar is not None:
            ax2.plot(
                idx,
                kappa_cvar.reindex(idx),
                color="#805ad5",
                linewidth=1.2,
                label="κ CVaR",
                linestyle="--",
            )
        ax2.set_ylabel("κ values")
        ax2.grid(False)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")
    else:
        ax.legend(loc="upper left")

    _add_caption(
        fig,
        caption
        or "Overlay of crisis_level (xai_prototypes_timeseries) vs cash ratio (holdings_timeseries); "
        "κ signals plotted when provided.",
    )
    return fig


def plot_cash_vs_crisis(
    crisis: pd.Series,
    cash_ratio: pd.Series,
    caption: str = "",
) -> Figure:
    fig, ax = _new_figure()
    data = pd.DataFrame({"crisis": crisis, "cash": cash_ratio}).dropna()
    hb = ax.hexbin(
        data["crisis"],
        data["cash"],
        gridsize=30,
        cmap="Blues",
        mincnt=1,
    )
    ax.set_xlabel("Crisis level")
    ax.set_ylabel("Cash ratio")
    ax.set_title("Cash deployment vs crisis probability")
    ax.figure.colorbar(hb, ax=ax, label="Count")
    _add_caption(
        fig,
        caption
        or "Hexbin density of crisis_level (xai_prototypes_timeseries) against cash ratio (holdings_timeseries).",
    )
    return fig


def plot_turnover_vs_reward(
    turnover: pd.Series,
    rewards: pd.Series,
    caption: str = "",
) -> Figure:
    fig, ax = _new_figure()
    data = pd.DataFrame({"turnover": turnover, "reward": rewards}).dropna()
    ax.scatter(data["turnover"], data["reward"] * 100, alpha=0.6, color="#4fd1c5")
    ax.set_xlabel("Turnover (abs qty * price)")
    ax.set_ylabel("Return (%)")
    ax.set_title("Turnover vs return per step")
    ax.grid(alpha=0.2)
    if not data.empty:
        corr = data["turnover"].corr(data["reward"])
        ax.text(0.02, 0.95, f"Pearson corr = {corr:.2f}", transform=ax.transAxes, va="top")
    _add_caption(
        fig,
        caption
        or "Turnover aggregated from trades.csv (abs(qty*price) per step) vs per_step_returns.",
    )
    return fig


def plot_cost_footprint(daily_cost: pd.Series, caption: str = "") -> Figure:
    fig, ax = _new_figure()
    idx = daily_cost.index
    ax.bar(idx, daily_cost, color="#f56565", alpha=0.7, label="Daily transaction cost")
    cumulative = daily_cost.cumsum()
    ax2 = ax.twinx()
    ax2.plot(idx, cumulative, color="#c53030", linewidth=1.5, label="Cumulative cost")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cost")
    ax2.set_ylabel("Cumulative cost")
    _format_time_axis(ax, idx)
    ax.grid(alpha=0.2)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    _add_caption(
        fig,
        caption or "Transaction costs sourced from trades.csv (tx_cost column); cumulative overlay highlights footprint.",
    )
    return fig


def plot_proto_area(
    topk_weights: pd.DataFrame,
    caption: str = "",
) -> Figure:
    fig, ax = _new_figure()
    idx = topk_weights.index
    ax.stackplot(idx, topk_weights.T, labels=topk_weights.columns, alpha=0.85)
    _format_time_axis(ax, idx)
    ax.set_ylabel("Weight")
    ax.set_xlabel("Step")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.set_title("Prototype weight top-K mix")
    _add_caption(
        fig,
        caption
        or "Prototype weight composition derived from xai_prototypes_timeseries topk_{i}_weight fields (other = residual).",
    )
    return fig


def plot_entropy_temp(
    proto_entropy: pd.Series,
    action_temp: Optional[pd.Series] = None,
    alpha_c: Optional[pd.Series] = None,
    caption: str = "",
) -> Figure:
    fig, ax = _new_figure()
    idx = proto_entropy.index
    ax.plot(idx, proto_entropy, color="#2b6cb0", linewidth=1.5, label="Prototype entropy")
    if action_temp is not None:
        ax.plot(idx, action_temp.reindex(idx), color="#dd6b20", linewidth=1.2, label="Action temperature")
    if alpha_c is not None:
        ax.plot(idx, alpha_c.reindex(idx), color="#38a169", linewidth=1.2, linestyle="--", label="α_c mean")
    _format_time_axis(ax, idx)
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    ax.set_title("Uncertainty signals (entropy / temperature / α)")
    _add_caption(
        fig,
        caption
        or "Signals pulled from xai_prototypes_timeseries (proto_entropy, action_temp, alpha_c_mean) highlight exploration vs exploitation.",
    )
    return fig


def plot_attr_heatmap(
    attr_matrix: pd.DataFrame,
    feature_order: Optional[List[str]] = None,
    regime_marks: Optional[pd.Series] = None,
    normalisation_note: str = "",
    caption: str = "",
) -> Figure:
    fig, ax = _new_figure()
    matrix = attr_matrix.copy()
    if feature_order:
        existing = [f for f in feature_order if f in matrix.columns]
        remaining = [f for f in matrix.columns if f not in existing]
        matrix = matrix[existing + remaining]
    im = ax.imshow(
        matrix.values,
        aspect="auto",
        cmap="RdBu",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xlabel("Features")
    ax.set_ylabel("Sampled steps")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    fig.colorbar(im, ax=ax, label="Normalised attribution")
    if regime_marks is not None and not regime_marks.empty:
        crisis_steps = matrix.index.intersection(regime_marks[regime_marks > 0.5].index)
        for step in crisis_steps:
            idx = matrix.index.get_loc(step)
            ax.axhline(idx - 0.5, color="#cbd5e0", linewidth=0.5)
            ax.axhline(idx + 0.5, color="#cbd5e0", linewidth=0.5)
    note = normalisation_note or "Rows normalised so that Σ|attr|=1 per step."
    _add_caption(
        fig,
        caption or f"Feature attributions (xai_feature_attributions) heatmap; {note} N={matrix.shape[0]} samples.",
    )
    return fig


def plot_attr_regime_bars(
    topk_normal: Dict[str, float],
    topk_crisis: Dict[str, float],
    caption: str = "",
) -> Figure:
    fig, axes = _new_figure(nrows=1, ncols=2, sharex=False)
    axes = np.atleast_1d(axes)

    def _plot(ax: Axes, data: Dict[str, float], title: str) -> None:
        if not data:
            ax.set_axis_off()
            ax.set_title(f"{title}\n(no data)")
            return
        features = list(data.keys())
        values = np.array(list(data.values()))
        ax.barh(features, values, color="#4a5568")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlabel("Normalised attribution")

    _plot(axes[0], topk_normal, "Normal regime")
    _plot(axes[1], topk_crisis, "Crisis regime")
    _add_caption(
        fig,
        caption
        or "Top-K feature importances from xai_summary.json comparing normal vs crisis regimes (values already normalised).",
    )
    return fig


def _convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    return obj


def save_evaluation_results(
    results: Dict,
    output_dir: Path,
    config: Optional[Dict] = None,
) -> None:
    """
    Persist evaluation artefacts in the new XAI-friendly layout.

    - evaluation_results.json: full payload (metrics, configs, arrays)
    - evaluation_insights.json: lightweight summary consumed by viz CLI
    - holdings_timeseries.csv: per-step asset/cash weights (optional)
    - trades.csv: reconstructed trade log (optional)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def _to_float_array(values) -> np.ndarray:
        if not _is_non_empty(values):
            return np.array([], dtype=np.float64)
        return np.asarray(values, dtype=np.float64).reshape(-1)

    timestamp = datetime.now(timezone.utc).isoformat()
    series = results.get("series")
    if series is None:
        series = {}
        for key in (
            "portfolio_values",
            "per_step_returns",
            "per_step_returns_raw",
            "value_returns",
            "value_returns_raw",
            "transaction_costs_series",
            "turnover_executed_series",
            "turnover_target_series",
            "cash_ratio_series",
            "dates",
        ):
            if key in results:
                series[key] = results[key]
        results["series"] = series

    tables = results.get("tables")
    if tables is None:
        tables = {}
        for key in ("holdings_timeseries", "trades"):
            if key in results:
                tables[key] = results[key]
        results["tables"] = tables

    paths = results.get("paths")
    if paths is None:
        paths = {}
        results["paths"] = paths

    holdings_records = tables.get("holdings_timeseries")
    if isinstance(holdings_records, list) and _is_non_empty(holdings_records):
        holdings_df = pd.DataFrame(holdings_records)
        if "step" in holdings_df.columns:
            holdings_df = holdings_df.sort_values("step")
        holdings_file = "holdings_timeseries.csv"
        holdings_df.to_csv(output_path / holdings_file, index=False)
        paths["holdings_csv"] = holdings_file
        tables["holdings_timeseries"] = {"rows": int(holdings_df.shape[0])}
        cash_series_from_table = holdings_df.get("CASH")
    else:
        cash_series_from_table = None
        if "holdings_csv" not in paths:
            candidate = output_path / "holdings_timeseries.csv"
            if candidate.is_file():
                paths["holdings_csv"] = candidate.name

    trade_records = tables.get("trades")
    if isinstance(trade_records, list) and _is_non_empty(trade_records):
        trades_df = pd.DataFrame(trade_records)
        columns = ["timestamp", "step", "ticker", "qty", "price", "tx_cost", "side"]
        trades_df = trades_df.reindex(columns=[col for col in columns if col in trades_df.columns])
        trades_df = trades_df.sort_values(["step", "ticker"]) if "step" in trades_df.columns else trades_df
        trades_file = "trades.csv"
        trades_df.to_csv(output_path / trades_file, index=False)
        paths["trades_csv"] = trades_file
        tables["trades"] = {"rows": int(trades_df.shape[0])}
    else:
        if "trades_csv" not in paths:
            candidate = output_path / "trades.csv"
            if candidate.is_file():
                paths["trades_csv"] = candidate.name

    metrics = results.get("metrics", {}) or {}
    series_portfolio = _to_float_array(series.get("portfolio_values"))
    returns_arr = _to_float_array(series.get("per_step_returns"))
    cash_series = _to_float_array(series.get("cash_ratio_series"))
    if cash_series.size == 0 and cash_series_from_table is not None:
        cash_series = cash_series_from_table.to_numpy(dtype=float)

    total_return = metrics.get("total_return")
    if total_return is None and series_portfolio.size > 1:
        total_return = float(series_portfolio[-1] / series_portfolio[0] - 1.0)

    final_value = metrics.get("final_value")
    if final_value is None and series_portfolio.size:
        final_value = float(series_portfolio[-1])

    volatility = metrics.get("volatility")
    if volatility is None and returns_arr.size:
        volatility = float(np.std(returns_arr) * np.sqrt(252))

    avg_cash = float(np.nanmean(cash_series)) if cash_series.size else None
    median_cash = float(np.nanmedian(cash_series)) if cash_series.size else None
    min_cash = float(np.nanmin(cash_series)) if cash_series.size else None
    max_cash = float(np.nanmax(cash_series)) if cash_series.size else None

    crisis_series = _to_float_array(series.get("crisis_levels"))
    if crisis_series.size == 0 and "irt" in results:
        crisis_series = _to_float_array(results["irt"].get("crisis_levels"))
    crisis_pct = metrics.get("crisis_regime_pct")
    if crisis_pct is None and crisis_series.size:
        crisis_pct = float(np.mean(crisis_series >= 0.5))

    cleaned_dates = []
    dates_raw = series.get("dates") or []
    if dates_raw:
        cleaned_dates = [None if d in (None, "", "None") else str(d) for d in dates_raw]
        if cleaned_dates and all(d is None for d in cleaned_dates):
            cleaned_dates = []

    period_start = cleaned_dates[0] if cleaned_dates else None
    period_end = cleaned_dates[-1] if cleaned_dates else None

    insights = {
        "timestamp": timestamp,
        "total_return": total_return,
        "annualized_return": metrics.get("annualized_return"),
        "volatility": volatility,
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "sortino_ratio": metrics.get("sortino_ratio"),
        "max_drawdown": metrics.get("max_drawdown"),
        "cvar_5": metrics.get("cvar_5"),
        "final_value": final_value,
        "steps": metrics.get("n_steps"),
        "crisis_regime_pct": crisis_pct,
        "crisis_activation_rate": metrics.get("crisis_activation_rate"),
        "avg_cash_weight": avg_cash,
        "median_cash_weight": median_cash,
        "cash_weight_min": min_cash,
        "cash_weight_max": max_cash,
        "avg_turnover_executed": metrics.get("avg_turnover"),
        "avg_turnover_target": metrics.get("avg_turnover_target"),
        "files": paths.copy(),
        "period_start": period_start,
        "period_end": period_end,
    }

    (output_path / "evaluation_insights.json").write_text(
        json.dumps(insights, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    serialisable_results = _convert_to_serializable(results)
    payload = {
        "timestamp": timestamp,
        "config": _convert_to_serializable(config) if config is not None else None,
        "results": serialisable_results,
    }
    (output_path / "evaluation_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def plot_all(
    portfolio_values: Optional[Sequence[float]],
    returns: Optional[Sequence[float]],
    output_dir: str,
    benchmark_returns: Optional[Sequence[float]] = None,
    metrics: Optional[Dict[str, float]] = None,
    **_,  # ignore legacy kwargs
) -> None:
    """
    Backwards-compatible helper used by legacy training scripts.

    Generates the three core plots (equity, drawdown, return distribution)
    in ``output_dir`` and emits a deprecation warning directing callers to the
    new reporting pipeline.
    """
    warnings.warn(
        "plot_all is deprecated; invoke scripts/visualize_from_json.py for the new XAI dashboard.",
        DeprecationWarning,
        stacklevel=2,
    )
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    values_arr = (
        np.asarray(portfolio_values, dtype=float).reshape(-1)
        if portfolio_values is not None
        else np.array([], dtype=float)
    )
    returns_arr = (
        np.asarray(returns, dtype=float).reshape(-1)
        if returns is not None
        else np.array([], dtype=float)
    )
    if returns_arr.size == 0 and values_arr.size >= 2:
        prev = np.clip(values_arr[:-1], 1e-8, None)
        returns_arr = (values_arr[1:] - values_arr[:-1]) / prev

    index = np.arange(max(len(values_arr), len(returns_arr)))
    if values_arr.size > 0:
        equity = pd.Series(values_arr / max(values_arr[0], 1e-8), index=np.arange(values_arr.size))
    else:
        equity = compute_equity_curve(returns_arr, index=index[: returns_arr.size])
    drawdown = compute_drawdown(equity)
    returns_series = pd.Series(returns_arr, index=np.arange(returns_arr.size))

    benchmark = None
    if benchmark_returns is not None:
        bench_arr = np.asarray(benchmark_returns, dtype=float).reshape(-1)
        if bench_arr.size:
            benchmark = compute_equity_curve(
                bench_arr,
                base_value=1.0,
                index=np.arange(min(len(bench_arr), len(equity))),
            )

    annotation = metrics or {}
    fig = plot_equity(equity, benchmark=benchmark, annotations=annotation)
    fig.savefig(out_path / "legacy_equity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig = plot_drawdown(drawdown)
    fig.savefig(out_path / "legacy_drawdown.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    if not returns_series.empty:
        fig = plot_returns_dist(returns_series, cvar_5=annotation.get("CVaR@5%"))
        fig.savefig(out_path / "legacy_returns.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


__all__ = [
    "sanitize_returns",
    "compute_equity_curve",
    "compute_drawdown",
    "compute_rolling_stats",
    "normalise_attributions",
    "resample_matrix",
    "ensure_datetime_index",
    "plot_equity",
    "plot_drawdown",
    "plot_returns_dist",
    "plot_rolling_risk_return",
    "plot_crisis_kappa_cash",
    "plot_cash_vs_crisis",
    "plot_turnover_vs_reward",
    "plot_cost_footprint",
    "plot_proto_area",
    "plot_entropy_temp",
    "plot_attr_heatmap",
    "plot_attr_regime_bars",
    "plot_all",
    "save_evaluation_results",
]
