# scripts/train_irt.py
# SAC와 IRT 정책을 학습·평가하는 명령행 유틸리티를 제공한다.

"""
IRT Policy 학습 스크립트

위기 적응형 포트폴리오 관리를 위해 SAC와 IRTPolicy를 결합한 학습·평가 파이프라인을 제공한다.

사용법:
    # 기본 설정으로 학습 및 평가
    python scripts/train_irt.py --mode both

    # 커스텀 alpha로 학습만 수행
    python scripts/train_irt.py --mode train --alpha 0.5 --episodes 200

    # 저장된 모델 평가
    python scripts/train_irt.py --mode eval --checkpoint logs/irt/.../irt_final.zip
"""

import argparse
import json
import os
import random
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from finrl.config import (
    INDICATORS,
    SAC_PARAMS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
)
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch
from stable_baselines3.common.logger import (
    HumanOutputFormat,
    Logger,
    TensorBoardOutputFormat,
)

# Import IRT Policy
from finrl.agents.irt import IRTPolicy

# Import evaluation function
import sys

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import evaluate_model
from finrl.agents.stablebaselines3 import StrictEvalCallback


def set_global_seed(seed: int) -> None:
    """
    주요 난수 생성기(PyTorch, NumPy 등)에 동일한 시드를 설정해 재현성을 확보한다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_environment(env, seed: int) -> None:
    """
    지원되는 경우 환경과 내부 공간 전체에 시드를 주입한다.
    """
    target_envs: Iterable[Any]
    if hasattr(env, "envs") and hasattr(env, "seed"):
        num_envs = len(getattr(env, "envs", []))
        env.seed(seed)
        target_envs = getattr(env, "envs")
    else:
        if hasattr(env, "seed"):
            env.seed(seed)
        elif hasattr(env, "_seed"):  # legacy gym
            env._seed(seed)
        target_envs = (env,)
    if hasattr(env, "reset"):
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()

    for idx, sub_env in enumerate(target_envs):
        sub_seed = seed + idx * 997
        if hasattr(sub_env, "reset"):
            try:
                sub_env.reset(seed=sub_seed)
            except TypeError:
                sub_env.reset()
        action_space = getattr(sub_env, "action_space", None)
        if action_space is not None and hasattr(action_space, "seed"):
            action_space.seed(sub_seed)
        observation_space = getattr(sub_env, "observation_space", None)
        if observation_space is not None and hasattr(observation_space, "seed"):
            observation_space.seed(sub_seed)


def _unwrap_env(env):
    """
    VecEnv 및 Gym 래퍼를 재귀적으로 풀어 실제 기본 환경을 반환한다.
    """
    env_ref = env
    visited = set()
    while True:
        if id(env_ref) in visited:
            break
        visited.add(id(env_ref))
        if hasattr(env_ref, "envs") and len(getattr(env_ref, "envs")) > 0:
            env_ref = env_ref.envs[0]
            continue
        if hasattr(env_ref, "env"):
            env_ref = env_ref.env
            continue
        break
    return env_ref


ALLOWED_TENSORBOARD_TAGS = {
    "returns/episode_sharpe",
    "returns/episode_sortino",
    "returns/annualized_return",
    "risk/max_drawdown",
    "risk/cvar_p05",
    "risk/crisis_level_avg",
    "risk/kappa_sharpe",
    "risk/kappa_cvar",
    "reward/log_return",
    "reward/sharpe_term",
    "reward/cvar_term",
    "reward/total",
    "action/turnover_executed",
    "action/cash_weight",
    "train/actor_loss",
    "train/critic_loss",
    "train/entropy",
    "train/alpha",
    "xai/proto_entropy",
}


class TensorboardWhitelistOutputFormat(TensorBoardOutputFormat):
    """
    허용된 FinFlow 텔레메트리 항목만 기록하도록 필터링하는 TensorBoard 출력기
    """

    def __init__(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        super().__init__(folder)

    def _is_allowed(self, key: str) -> bool:
        return key in ALLOWED_TENSORBOARD_TAGS

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        filtered_keys = {}
        filtered_excluded = {}
        for k, v in key_values.items():
            if self._is_allowed(k):
                filtered_keys[k] = v
                filtered_excluded[k] = key_excluded.get(k, tuple())

        if not filtered_keys:
            return

        super().write(filtered_keys, filtered_excluded, step)


class IRTLoggingCallback(BaseCallback):
    """Minimal TensorBoard logging for FinFlow IRT training."""

    _METRIC_KEYS = {
        "reward/log_return": "reward_log_return",
        "reward/sharpe_term": ("reward_components", "sharpe_term"),
        "reward/cvar_term": ("reward_components", "cvar_term"),
        "reward/total": "reward_total",
        "risk/cvar_p05": "cvar_value",
        "risk/crisis_level_avg": "crisis_level",
        "risk/kappa_sharpe": "kappa_sharpe",
        "risk/kappa_cvar": "kappa_cvar",
        "action/turnover_executed": "turnover_executed",
        "action/cash_weight": "cash_weight",
    }

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = max(1, int(log_freq))
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self._metrics = {key: [0.0, 0] for key in self._METRIC_KEYS}

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(val):
            return None
        return val

    def _accumulate_metric(self, key: str, value) -> None:
        val = self._to_float(value)
        if val is None:
            return
        bucket = self._metrics[key]
        bucket[0] += val
        bucket[1] += 1

    def _value_from_info(self, info: dict, descriptor):
        if isinstance(descriptor, tuple):
            container = info.get(descriptor[0]) or {}
            return container.get(descriptor[1])
        return info.get(descriptor)

    def _on_training_start(self) -> None:
        self._reset_buffers()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") if isinstance(self.locals, dict) else None
        if infos:
            for info in infos:
                if not info:
                    continue
                for tb_key, desc in self._METRIC_KEYS.items():
                    value = self._value_from_info(info, desc)
                    self._accumulate_metric(tb_key, value)
        if self.n_calls % self.log_freq == 0:
            self._flush()
        return True

    def _on_training_end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        for key, (total, count) in self._metrics.items():
            if count:
                self.logger.record(key, total / count)
        proto_entropy = self._compute_proto_entropy()
        if proto_entropy is not None:
            self.logger.record("xai/proto_entropy", proto_entropy)
        alpha = self._current_alpha()
        if alpha is not None:
            self.logger.record("train/alpha", alpha)
        self._reset_buffers()

    def _compute_proto_entropy(self) -> Optional[float]:
        model = getattr(self, "model", None)
        if model is None:
            return None
        policy = getattr(model, "policy", None)
        if policy is None or not hasattr(policy, "get_irt_info"):
            return None
        try:
            info = policy.get_irt_info()
        except AttributeError:
            return None
        if not info:
            return None
        weights = info.get("w")
        if weights is None:
            return None
        if isinstance(weights, torch.Tensor):
            tensor = weights.detach().float()
        else:
            tensor = torch.as_tensor(weights, dtype=torch.float32)
        if tensor.ndim < 2:
            return None
        probs = torch.clamp(tensor, min=1e-8)
        entropy = -(probs * probs.log()).sum(dim=-1).mean()
        if not torch.isfinite(entropy):
            return None
        return float(entropy.item())

    def _current_alpha(self) -> Optional[float]:
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "log_ent_coef"):
            return None
        log_ent_coef = getattr(model, "log_ent_coef")
        if log_ent_coef is None:
            return None
        if isinstance(log_ent_coef, torch.Tensor):
            value = torch.exp(log_ent_coef.detach()).cpu().item()
        elif isinstance(log_ent_coef, (float, int)):
            value = float(math.exp(log_ent_coef))
        else:
            return None
        if not np.isfinite(value):
            return None
        return float(value)


class ActionTempSchedulerCallback(BaseCallback):
    """Linearly anneal the IRT action temperature during training."""

    def __init__(self, final_temp: float, anneal_steps: int):
        super().__init__()
        self.final_temp = float(final_temp)
        self.anneal_steps = max(1, int(anneal_steps))
        self.initial_temp: Optional[float] = None

    def _get_policy(self):
        return getattr(self.model, "policy", None)

    def _sync_temp(self, value: float) -> None:
        policy = self._get_policy()
        if policy is None:
            return
        value = float(value)
        setattr(policy, "action_temp", value)
        actor = getattr(policy, "actor", None)
        if actor is not None:
            if hasattr(actor, "action_temp"):
                actor.action_temp = value
            irt_actor = getattr(actor, "irt_actor", None)
            if irt_actor is not None and hasattr(irt_actor, "action_temp"):
                irt_actor.action_temp = value

    def _on_training_start(self) -> None:
        policy = self._get_policy()
        if policy is None:
            return
        current = getattr(policy, "action_temp", None)
        if current is not None:
            self.initial_temp = float(current)

    def _on_step(self) -> bool:
        if self.initial_temp is None:
            return True
        progress = min(self.model.num_timesteps, self.anneal_steps) / self.anneal_steps
        new_temp = self.initial_temp + progress * (self.final_temp - self.initial_temp)
        self._sync_temp(new_temp)
        return True


class CrisisBridgeCallback(BaseCallback):
    """
    정책이 산출한 위기 레벨을 환경의 위험 보상 모듈에 전달하는 브리지 콜백.

    동작:
    1. 정책의 `get_irt_info()`로부터 `crisis_level`을 추출합니다.
    2. 환경의 `risk_reward.set_crisis_level()`을 호출해 보상 모듈의 위기 상태를 동기화합니다.
    3. 동기화된 위기 레벨을 기반으로 κ(c) 등 위험 민감 파라미터가 업데이트됩니다.

    사용 조건:
    - `reward_type='adaptive_risk'` 환경에서만 의미가 있습니다.
    - IRT Policy를 사용하는 평가/학습 루프에서 활성화합니다.
    """

    def __init__(self, learning_starts: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.learning_starts = learning_starts
        self._warned_once = False
        self._connected_once = False

    def _on_step(self) -> bool:
        if self.model is None:
            return True

        # 초기 스텝부터 위기 레벨이 산출되므로 별도의 워밍업 없이 즉시 동기화한다.

        info = None
        policy = getattr(self.model, "policy", None)
        if policy is not None and hasattr(policy, "get_irt_info"):
            try:
                info = policy.get_irt_info()
            except AttributeError as exc:
                if not self._warned_once:
                    print(f"⚠️  CrisisBridgeCallback: Failed to fetch IRT info ({exc})")
                    self._warned_once = True
                return True

        if not info or "crisis_level" not in info:
            if not self._warned_once:
                print("⚠️  CrisisBridgeCallback: IRT info not available, skipping")
                self._warned_once = True
            return True

        crisis_tensor = info["crisis_level"]
        if isinstance(crisis_tensor, torch.Tensor):
            crisis_level = float(crisis_tensor.detach().cpu().mean().item())
        elif isinstance(crisis_tensor, (np.ndarray, list)):
            crisis_level = float(np.mean(crisis_tensor))
        else:
            crisis_level = float(crisis_tensor)

        base_env = _unwrap_env(self.model.get_env())
        setattr(base_env, "_crisis_level", crisis_level)

        risk_reward = getattr(base_env, "risk_reward", None)
        if risk_reward is not None and hasattr(risk_reward, "set_crisis_level"):
            try:
                risk_reward.set_crisis_level(crisis_level)
            except (TypeError, ValueError) as exc:
                if not self._warned_once:
                    print(
                        f"⚠️  CrisisBridgeCallback: Failed to update crisis level ({exc})"
                    )
                    self._warned_once = True
                return True

        if hasattr(base_env, "_crisis_history"):
            base_env._crisis_history.append(
                (int(self.model.num_timesteps), crisis_level)
            )

        if not self._connected_once:
            print(
                f"✅ CrisisBridgeCallback: Successfully connected (t={self.model.num_timesteps})"
            )
            self._connected_once = True
            self._warned_once = False

        return True


def create_env(
    df,
    stock_dim,
    tech_indicators,
    reward_type="basic",
    lambda_dsr=0.1,
    lambda_cvar=0.05,
    use_weighted_action: bool = True,
    weight_slippage: float = 0.001,
    weight_transaction_cost: float = 0.0005,
    reward_scaling: float = 1e-4,
    adaptive_lambda_sharpe: float = 0.20,
    adaptive_lambda_cvar: float = 0.40,
    adaptive_lambda_turnover: float = 0.0,
    adaptive_crisis_gain_sharpe: float = -0.15,
    adaptive_crisis_gain_cvar: float = 0.25,
    adaptive_dsr_beta: float = 0.92,
    adaptive_cvar_window: int = 40,
):
    """DSR + CVaR 보상을 지원하는 StockTradingEnv를 생성한다."""

    # 상태 = 잔고(1) + 가격(N) + 보유량(N) + 기술지표(K*N)
    # `reward_type='dsr_cvar'`일 때는 DSR·CVaR 보조 지표가 추가되어 차원이 2만큼 증가한다.
    state_space = 1 + (len(tech_indicators) + 2) * stock_dim

    env_kwargs = {
        "df": df,
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": reward_scaling,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": tech_indicators,
        "print_verbosity": 500,
        # 리스크 민감 보상 구성 요소
        "reward_type": reward_type,
        "lambda_dsr": lambda_dsr,
        "lambda_cvar": lambda_cvar,
        "use_weighted_action": use_weighted_action,
        "weight_slippage": weight_slippage,
        "weight_transaction_cost": weight_transaction_cost,
        "adaptive_lambda_sharpe": adaptive_lambda_sharpe,
        "adaptive_lambda_cvar": adaptive_lambda_cvar,
        "adaptive_lambda_turnover": adaptive_lambda_turnover,
        "adaptive_crisis_gain_sharpe": adaptive_crisis_gain_sharpe,
        "adaptive_crisis_gain_cvar": adaptive_crisis_gain_cvar,
        "adaptive_dsr_beta": adaptive_dsr_beta,
        "adaptive_cvar_window": adaptive_cvar_window,
    }

    return StockTradingEnv(**env_kwargs)


@dataclass
class DiversificationConfig:
    start_index: int
    start_date: str
    end_index: int
    end_date: str
    window_days: int
    tx_scale: float
    slippage_scale: float
    row_count: int

    def to_meta(self) -> Dict[str, Any]:
        return {
            "start_index": int(self.start_index),
            "start_date": self.start_date,
            "end_index": int(self.end_index),
            "end_date": self.end_date,
            "window_days": int(self.window_days),
            "tx_scale": float(self.tx_scale),
            "slippage_scale": float(self.slippage_scale),
            "row_count": int(self.row_count),
        }


def _sorted_unique_dates(df) -> List[Any]:
    dates = df["date"].unique()
    return sorted(dates.tolist())


def _slice_df_by_dates(df, start_date, end_date) -> Tuple[Any, Any, pd.DataFrame]:
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    window = df.loc[mask].copy()
    if window.empty:
        raise ValueError("Selected diversification window is empty.")
    window.sort_values(by=["date", "tic"], inplace=True)
    day_codes = pd.factorize(window["date"], sort=False)[0]
    window.index = day_codes
    window.index.name = None
    return start_date, end_date, window


def _sample_scales(rng: np.random.Generator, magnitude: float) -> float:
    if magnitude <= 0:
        return 1.0
    sample = float(rng.uniform(-magnitude, magnitude))
    scale = 1.0 + sample
    return max(scale, 0.05)


def _choose_offsets(
    mode: str,
    n_envs: int,
    start_mode: str,
    max_start: int,
    rng: np.random.Generator,
    window_step: int,
) -> List[int]:
    if mode == "off":
        return [0]
    total = max(1, n_envs)
    if max_start <= 0:
        return [0 for _ in range(total)]
    if start_mode == "random":
        return [int(rng.integers(0, max_start + 1)) for _ in range(total)]
    if start_mode == "uniform":
        if total == 1:
            return [int(max_start // 2)]
        lin = np.linspace(0, max_start, num=total)
        return [int(round(v)) for v in lin]
    if start_mode == "rolling":
        step = max(1, min(window_step, max_start if max_start > 0 else window_step))
        offsets: List[int] = []
        cursor = 0
        for _ in range(total):
            offsets.append(min(cursor, max_start))
            cursor += step
            if cursor > max_start:
                cursor = cursor % (max_start + 1)
        return offsets
    raise ValueError(f"Unsupported start_offset_mode: {start_mode}")


def build_training_env_with_diversification(
    df,
    stock_dim: int,
    tech_indicators: Sequence[str],
    args,
) -> Tuple[Any, Dict[str, Any]]:
    mode = args.env_diversify
    rolling_offsets = args.start_offset_mode == "rolling"
    rng = np.random.default_rng(args.seed + 17041)
    unique_dates = _sorted_unique_dates(df)
    if not unique_dates:
        raise ValueError("Training dataframe does not contain any dates.")
    total_days = len(unique_dates)
    min_window = max(1, min(total_days, max(32, int(0.1 * total_days))))
    window_step = max(1, min(int(args.window_step), total_days))

    if rolling_offsets:
        window_days = max(min_window, window_step)
        max_start = max(0, total_days - window_days)
    else:
        window_days = total_days
        max_start = max(0, total_days - min_window)

    n_envs = 1 if mode != "vector" else max(1, int(args.n_envs))
    offsets = _choose_offsets(
        mode=mode,
        n_envs=n_envs,
        start_mode=args.start_offset_mode,
        max_start=max_start,
        rng=rng,
        window_step=window_step,
    )

    configs: List[DiversificationConfig] = []
    env_fns: List[Any] = []

    base_tx = 0.0005
    base_slippage = 0.001

    for offset in offsets:
        start_idx = max(0, min(offset, total_days - 1))
        if rolling_offsets:
            end_idx = min(total_days, start_idx + window_days)
        else:
            end_idx = total_days
        end_idx = max(start_idx + 1, end_idx)
        start_date = unique_dates[start_idx]
        end_date = unique_dates[end_idx - 1]
        _, _, window_df = _slice_df_by_dates(df, start_date, end_date)
        tx_scale = (
            1.0 if mode == "off" else _sample_scales(rng, float(args.domain_rand_tx))
        )
        slip_scale = (
            1.0
            if mode == "off"
            else _sample_scales(rng, float(args.domain_rand_slippage))
        )
        config = DiversificationConfig(
            start_index=int(start_idx),
            start_date=str(start_date),
            end_index=int(end_idx - 1),
            end_date=str(end_date),
            window_days=int(end_idx - start_idx),
            tx_scale=float(tx_scale),
            slippage_scale=float(slip_scale),
            row_count=int(window_df.shape[0]),
        )
        configs.append(config)

        env_kwargs = {
            "df": window_df,
            "stock_dim": stock_dim,
            "tech_indicators": tech_indicators,
            "reward_type": args.reward_type,
            "lambda_dsr": args.lambda_dsr,
            "lambda_cvar": args.lambda_cvar,
            "reward_scaling": args.reward_scale,
            "adaptive_lambda_sharpe": args.adaptive_lambda_sharpe,
            "adaptive_lambda_cvar": args.adaptive_lambda_cvar,
            "adaptive_lambda_turnover": args.adaptive_lambda_turnover,
            "adaptive_crisis_gain_sharpe": args.adaptive_crisis_gain_sharpe,
            "adaptive_crisis_gain_cvar": args.adaptive_crisis_gain_cvar,
            "adaptive_dsr_beta": args.adaptive_dsr_beta,
            "adaptive_cvar_window": args.adaptive_cvar_window,
            "use_weighted_action": True,
            "weight_slippage": base_slippage * slip_scale,
            "weight_transaction_cost": base_tx * tx_scale,
        }

        def _make_env(env_kwargs=env_kwargs):
            return create_env(**env_kwargs)

        env_fns.append(_make_env)

    if mode == "vector" and len(env_fns) > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    effective_mode = mode
    if mode == "vector" and len(env_fns) == 1:
        effective_mode = "basic"

    diversify_meta: Dict[str, Any] = {
        "mode": effective_mode,
        "requested_mode": mode,
        "params": {
            "n_envs": len(env_fns),
            "start_offset_mode": args.start_offset_mode,
            "window_step": int(window_step),
            "min_window_days": int(min_window),
            "domain_rand_tx": float(args.domain_rand_tx),
            "domain_rand_slippage": float(args.domain_rand_slippage),
        },
        "configs": [cfg.to_meta() for cfg in configs],
    }

    return vec_env, diversify_meta


def train_irt(args):
    """IRT 모델 학습"""

    print("=" * 70)
    print("IRT Training - Dow Jones 30")
    print("=" * 70)

    set_global_seed(args.seed)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output, "irt", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    print("\n[Training Config]")
    print("  Model: SAC + IRT Policy")
    print(f"  Stocks: Dow Jones 30 ({len(DOW_30_TICKER)} tickers)")
    print(f"  Train period: {args.train_start} ~ {args.train_end}")
    print(f"  Test period: {args.test_start} ~ {args.test_end}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Initial alpha: {args.alpha}")
    print(f"  Output directory: {log_dir}")

    # 1. 데이터 다운로드
    print("\n[1/5] Downloading data...")
    df = YahooDownloader(
        start_date=args.train_start, end_date=args.test_end, ticker_list=DOW_30_TICKER
    ).fetch_data()
    print(f"  Downloaded: {df.shape[0]} rows")

    # 2. 피처 엔지니어링
    print("\n[2/5] Running feature engineering...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        user_defined_feature=False,
    )
    df_processed = fe.preprocess_data(df)
    print(f"  Feature columns: {df_processed.shape[1]}")

    # 3. 학습/평가 데이터 분할
    print("\n[3/5] Splitting train/test data...")
    train_df = data_split(df_processed, args.train_start, args.train_end)
    test_df = data_split(df_processed, args.test_start, args.test_end)
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")

    # 4. 환경 생성
    print("\n[4/5] Creating environments...")
    stock_dim = len(train_df.tic.unique())

    # 데이터 누락 경고
    if stock_dim != len(DOW_30_TICKER):
        removed_count = len(DOW_30_TICKER) - stock_dim
        print(f"  ⚠️ Warning: {removed_count} tickers excluded due to missing data")
        print(f"      (e.g., incomplete early-2008 listings such as Visa (V))")

    print(f"  Effective stock count: {stock_dim}")

    # 리스크 민감 보상 환경 생성
    train_env, diversify_meta = build_training_env_with_diversification(
        train_df,
        stock_dim,
        INDICATORS,
        args,
    )
    seed_environment(train_env, args.seed)
    base_train_env = _unwrap_env(train_env)
    print(
        f"  Train env: reward_scale={getattr(base_train_env, 'reward_scaling', None)}, "
        f"use_weighted_action={getattr(base_train_env, 'use_weighted_action', False)}, "
        f"slippage={getattr(base_train_env, 'weight_slippage', None)}, "
        f"tx_cost={getattr(base_train_env, 'weight_transaction_cost', None)}"
    )
    if diversify_meta["mode"] != "off":
        print(
            f"    Diversify: mode={diversify_meta['mode']}, "
            f"n_envs={diversify_meta['params']['n_envs']}, "
            f"window_step={diversify_meta['params']['window_step']}, "
            f"domain_rand_tx={diversify_meta['params']['domain_rand_tx']}, "
            f"domain_rand_slippage={diversify_meta['params']['domain_rand_slippage']}"
        )
        if diversify_meta.get("requested_mode") != diversify_meta["mode"]:
            print(
                "      (requested mode: {req}, downcast to {eff} because n_envs=1)".format(
                    req=diversify_meta.get("requested_mode"), eff=diversify_meta["mode"]
                )
            )
        for idx, cfg in enumerate(diversify_meta["configs"]):
            print(
                "      Env[{idx}] start={start} end={end} days={days} tx×{tx:.3f} slip×{sl:.3f}".format(
                    idx=idx,
                    start=cfg["start_date"],
                    end=cfg["end_date"],
                    days=cfg["window_days"],
                    tx=cfg["tx_scale"],
                    sl=cfg["slippage_scale"],
                )
            )

    tech_indicator_count = len(getattr(base_train_env, "tech_indicator_list", []))
    has_dsr_cvar = args.reward_type == "dsr_cvar"
    test_env = create_env(
        test_df,
        stock_dim,
        INDICATORS,
        reward_type=args.reward_type,
        lambda_dsr=args.lambda_dsr,
        lambda_cvar=args.lambda_cvar,
        reward_scaling=args.reward_scale,
        adaptive_lambda_sharpe=args.adaptive_lambda_sharpe,
        adaptive_lambda_cvar=args.adaptive_lambda_cvar,
        adaptive_lambda_turnover=args.adaptive_lambda_turnover,
        adaptive_crisis_gain_sharpe=args.adaptive_crisis_gain_sharpe,
        adaptive_crisis_gain_cvar=args.adaptive_crisis_gain_cvar,
        adaptive_dsr_beta=args.adaptive_dsr_beta,
        adaptive_cvar_window=args.adaptive_cvar_window,
    )
    seed_environment(test_env, args.seed)
    print(
        f"  테스트 환경 설정: reward_scale={test_env.reward_scaling}, "
        f"use_weighted_action={getattr(test_env, 'use_weighted_action', False)}, "
        f"slippage={getattr(test_env, 'weight_slippage', None)}, "
        f"tx_cost={getattr(test_env, 'weight_transaction_cost', None)}"
    )

    train_obs_dim = train_env.observation_space.shape[0]
    test_obs_dim = test_env.observation_space.shape[0]
    if train_obs_dim != test_obs_dim:
        raise ValueError(
            f"Train/Test observation space mismatch: train={train_obs_dim}, test={test_obs_dim}"
        )

    print(f"  State space: {getattr(base_train_env, 'state_space', train_obs_dim)}")
    print(f"  Action space: {base_train_env.action_space}")
    print(f"  Reward type: {args.reward_type}")
    if args.reward_type == "dsr_cvar":
        print(f"    λ_dsr: {args.lambda_dsr}, λ_cvar: {args.lambda_cvar}")
    elif args.reward_type == "adaptive_risk":
        rr = getattr(base_train_env, "risk_reward", None)
        lambda_sharpe = getattr(rr, "lambda_sharpe_base", None)
        lambda_cvar = getattr(rr, "lambda_cvar", None)
        lambda_turnover = getattr(rr, "lambda_turnover", None)
        crisis_gain_sharpe = getattr(rr, "crisis_gain_sharpe", None)
        crisis_gain_cvar = getattr(rr, "crisis_gain_cvar", None)
        print("    Reward configuration: adaptive_risk (Sharpe/CVaR)")
        print(
            f"    λ_sharpe: {lambda_sharpe}, λ_cvar: {lambda_cvar}, "
            f"λ_turnover: {lambda_turnover}, "
            f"g_S: {crisis_gain_sharpe}, g_C: {crisis_gain_cvar}"
        )

    # 5. IRT 모델 학습
    print("\n[5/5] Training SAC + IRT Policy...")

    # Callback 디렉토리 미리 생성 (FileNotFoundError 방지)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "best_model"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "eval"), exist_ok=True)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix="irt_model",
    )

    eval_callback = StrictEvalCallback(
        Monitor(test_env),
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    # IRT 관련 텔레메트리 콜백 등록
    irt_logging_callback = IRTLoggingCallback(log_freq=100, verbose=0)

    # IRT Policy kwargs
    effective_market_dim = (
        args.market_feature_dim
        if args.market_feature_dim is not None
        else 4 + tech_indicator_count
    )

    policy_kwargs = {
        "emb_dim": args.emb_dim,
        "m_tokens": args.m_tokens,
        "M_proto": args.M_proto,
        "alpha": args.alpha,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max if args.alpha_max else args.alpha,
        "ema_beta": args.ema_beta,
        "market_feature_dim": effective_market_dim,
        "stock_dim": stock_dim,
        "tech_indicator_count": tech_indicator_count,
        "has_dsr_cvar": has_dsr_cvar,
        # Dirichlet 농도 및 온도 파라미터는 평가 시점으로 전달된다.
        "dirichlet_min": args.dirichlet_min,
        "dirichlet_max": args.dirichlet_max,
        "action_temp": args.action_temp,
        # IRT Operator 파라미터
        "eps": args.eps,
        "max_iters": args.max_iters,
        "replicator_temp": args.replicator_temp,
        "eta_0": args.eta_0,
        "eta_1": args.eta_1,
        "gamma": args.gamma,
        "alpha_update_rate": args.alpha_update_rate,
        "alpha_feedback_gain": args.alpha_feedback_gain,
        "alpha_feedback_bias": args.alpha_feedback_bias,
        "directional_decay_min": args.directional_decay_min,
        "alpha_noise_std": args.alpha_noise_std,
        "alpha_crisis_source": args.alpha_crisis_source,
        # 위기 감지와 보상 모듈이 요구하는 신호 설정
        "w_r": args.w_r,
        "w_s": args.w_s,
        "w_c": args.w_c,
        # 위기 바이어스와 온도 보정 관련 설정 전달
        "eta_b": args.eta_b,
        "eta_b_min": args.eta_b_min,
        "eta_b_decay_steps": args.eta_b_decay_steps,
        "eta_T": args.eta_T,
        "p_star": args.p_star,
        "temperature_min": args.temperature_min,
        "temperature_max": args.temperature_max,
        "stat_momentum": args.stat_momentum,
        "eta_b_warmup_steps": args.eta_b_warmup_steps,
        "eta_b_warmup_value": args.eta_b_warmup_value,
        "crisis_guard_rate_init": args.crisis_guard_rate_init,
        "crisis_guard_rate_final": args.crisis_guard_rate_final,
        "crisis_guard_warmup_steps": args.crisis_guard_warmup_steps,
        "hysteresis_up": args.hysteresis_up,
        "hysteresis_down": args.hysteresis_down,
        "hysteresis_quantile": args.hysteresis_quantile,
        "k_s": args.k_s,
        "k_c": args.k_c,
        "k_b": args.k_b,
        "crisis_guard_rate": args.crisis_guard_rate,
    }

    # SAC parameters
    sac_params = SAC_PARAMS.copy()

    # 포트폴리오 탐색을 강화하기 위해 엔트로피 계수를 상향 조정한다.
    # 단순 연속 제어보다 큰 탐색 공간을 가지고 있으므로 0.5 수준이 안정적이었다.
    sac_params["ent_coef"] = 0.5  # 탐색 다양성을 위한 높은 엔트로피 계수
    sac_params["learning_starts"] = 1000  # SAC 워밍업 스텝을 줄여 빠르게 정책을 학습

    action_dim = int(train_env.action_space.shape[0])
    default_target_entropy = -float(action_dim)
    if args.target_entropy is not None:
        sac_params["target_entropy"] = float(args.target_entropy)
    elif args.target_entropy_mult is not None:
        sac_params["target_entropy"] = default_target_entropy * float(
            args.target_entropy_mult
        )

    if args.utd_ratio is not None and args.utd_ratio > 0:
        base_freq = sac_params.get("train_freq", 1)
        if isinstance(base_freq, tuple):
            freq_value = float(base_freq[0])
        else:
            freq_value = float(base_freq)
        gradient_steps = max(1, int(round(freq_value * float(args.utd_ratio))))
        sac_params["gradient_steps"] = gradient_steps

    # adaptive_risk 보상에서는 Critic 학습률을 약간 높여 민감도를 확보한다.
    if args.reward_type == "adaptive_risk":
        sac_params["learning_rate"] = 5e-4  # 1e-4 → 5e-4 (5배 증가)
        print("  Setting critic learning rate to 5e-4 to amplify crisis gradients")

    print(f"  SAC params: {sac_params}")
    print(f"  IRT params: {policy_kwargs}")

    # 위기 교량 콜백(CrisisBridgeCallback)은 adaptive_risk 환경에서만 활성화한다.
    crisis_bridge_callback = None
    if args.reward_type == "adaptive_risk":
        crisis_bridge_callback = CrisisBridgeCallback(
            learning_starts=0,  # 위기 레벨을 조기에 전달하기 위해 즉시 시작
            verbose=1,  # 브리지 동작을 디버깅하기 위해 verbose 활성화
        )
        print("  CrisisBridgeCallback enabled (policy → environment crisis signal)")

    sac_params["seed"] = args.seed

    # Create SAC model with IRT Policy
    model = SAC(
        policy=IRTPolicy,
        env=train_env,
        policy_kwargs=policy_kwargs,
        **sac_params,
        verbose=1,
        tensorboard_log=None,
    )

    tb_path = os.path.join(log_dir, "tensorboard")
    logger_formats = [
        HumanOutputFormat(sys.stdout),
        TensorboardWhitelistOutputFormat(tb_path),
    ]
    model.set_logger(Logger(log_dir, logger_formats))

    # 최적화 대상에 프로토타입, IRT 연산자, T-Cell이 모두 포함되었는지 확인한다.
    print("\n[Verification] Checking optimizer parameter groups...")
    actor_params = list(model.actor.parameters())
    actor_param_count = sum(p.numel() for p in actor_params)
    actor_requires_grad = sum(p.numel() for p in actor_params if p.requires_grad)
    print(f"  Actor total parameters: {actor_param_count:,}")
    print(f"  Trainable parameters: {actor_requires_grad:,}")

    # Prototype decoder 파라미터 확인
    if hasattr(model.actor, "irt_actor"):
        irt_actor = model.actor.irt_actor
        if hasattr(irt_actor, "decoders"):
            decoder_params = sum(
                p.numel() for d in irt_actor.decoders for p in d.parameters()
            )
            decoder_trainable = sum(
                p.numel()
                for d in irt_actor.decoders
                for p in d.parameters()
                if p.requires_grad
            )
            print(
                f"  Prototype decoder parameters: total {decoder_params:,} ({decoder_trainable:,} trainable)"
            )

            # Gradient flow 검증용 플래그
            if decoder_trainable == 0:
                print("  WARNING: all prototype decoder parameters are frozen!")
            elif decoder_trainable < decoder_params:
                print(
                    f"  WARNING: some prototype parameters are frozen ({decoder_params - decoder_trainable})"
                )

    # Total timesteps
    total_timesteps = 250 * args.episodes
    print(f"\n  Total timesteps: {total_timesteps}")
    print("  Starting training loop...")

    # Train
    callbacks = [checkpoint_callback, eval_callback, irt_logging_callback]
    if (
        args.action_temp_final is not None
        and args.action_temp_final != args.action_temp
    ):
        anneal_steps = args.action_temp_anneal_steps or total_timesteps
        callbacks.append(
            ActionTempSchedulerCallback(args.action_temp_final, anneal_steps)
        )
        print(
            f"  Action temperature scheduler: {args.action_temp} → {args.action_temp_final} over {anneal_steps} steps"
        )
    if crisis_bridge_callback is not None:
        callbacks.append(crisis_bridge_callback)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_model_path = os.path.join(log_dir, "irt_final.zip")
    model.save(final_model_path)

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"  Final model: {final_model_path}")
    print(f"  Best model: {os.path.join(log_dir, 'best_model', 'best_model.zip')}")
    print(f"  Logs: {log_dir}")

    env_meta = {
        "obs_dim": int(train_obs_dim),
        "action_dim": int(train_env.action_space.shape[0]),
        "feature_columns": df_processed.columns.tolist(),
        "tech_indicators": list(INDICATORS),
        "reward_type": args.reward_type,
        "use_weighted_action": getattr(base_train_env, "use_weighted_action", True),
        "weight_slippage": getattr(base_train_env, "weight_slippage", 0.001),
        "weight_transaction_cost": getattr(
            base_train_env, "weight_transaction_cost", 0.0005
        ),
        "reward_scaling": args.reward_scale,
        "stock_dim": stock_dim,
        "tech_indicator_count": tech_indicator_count,
        "has_dsr_cvar": has_dsr_cvar,
        "stat_momentum": args.stat_momentum,
        "adaptive_lambda_sharpe": args.adaptive_lambda_sharpe,
        "adaptive_lambda_cvar": args.adaptive_lambda_cvar,
        "adaptive_lambda_turnover": args.adaptive_lambda_turnover,
        "adaptive_crisis_gain_sharpe": args.adaptive_crisis_gain_sharpe,
        "adaptive_crisis_gain_cvar": args.adaptive_crisis_gain_cvar,
        "adaptive_dsr_beta": args.adaptive_dsr_beta,
        "adaptive_cvar_window": args.adaptive_cvar_window,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "env_diversify": diversify_meta["mode"],
        "env_diversify_requested": diversify_meta.get("requested_mode"),
        "env_diversify_params": diversify_meta["params"],
        "env_diversify_configs": diversify_meta["configs"],
    }
    with open(os.path.join(log_dir, "env_meta.json"), "w") as meta_fp:
        json.dump(env_meta, meta_fp, indent=2)

    return log_dir, final_model_path


def _log_returns_to_tensorboard(log_dir: str, metrics: Dict[str, Any]) -> None:
    candidates: list[str] = []
    if log_dir:
        candidates.append(os.path.join(log_dir, "tensorboard"))
        parent = (
            os.path.dirname(log_dir.rstrip(os.sep)) if log_dir.rstrip(os.sep) else ""
        )
        if parent and parent != log_dir:
            candidates.append(os.path.join(parent, "tensorboard"))

    tb_dir = next((path for path in candidates if os.path.isdir(path)), None)
    if not tb_dir:
        return

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return

    step = int(metrics.get("n_steps", 0) or 0)
    writer = SummaryWriter(tb_dir)
    try:
        sharpe = metrics.get("sharpe_ratio")
        if sharpe is not None:
            writer.add_scalar("returns/episode_sharpe", float(sharpe), step)

        sortino = metrics.get("sortino_ratio")
        if sortino is not None:
            writer.add_scalar("returns/episode_sortino", float(sortino), step)

        annual_return = metrics.get("annualized_return")
        if annual_return is not None:
            writer.add_scalar("returns/annualized_return", float(annual_return), step)

        max_dd = metrics.get("max_drawdown")
        if max_dd is not None:
            writer.add_scalar("risk/max_drawdown", float(max_dd), step)

        cvar_5 = metrics.get("cvar_5")
        if cvar_5 is not None:
            writer.add_scalar("risk/cvar_p05", float(cvar_5), step)

        writer.flush()
    finally:
        writer.close()


def test_irt(args, model_path=None):
    """IRT 모델 평가"""

    print("=" * 70)
    print("IRT Evaluation")
    print("=" * 70)

    set_global_seed(args.seed)

    # Model path
    if model_path is None:
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for --mode test")
        model_path = args.checkpoint

    print("\n[Evaluation Config]")
    print(f"  Model path: {model_path}")
    print(f"  Test period: {args.test_start} ~ {args.test_end}")

    # 메타데이터 로드 (관측 공간 검증 및 설정 유지)
    # 체크포인트에 저장된 환경 구성을 우선 적용한다.
    env_meta = {}
    expected_obs_dim = None
    use_weighted_action = True
    weight_slippage = 0.001
    weight_transaction_cost = 0.0005
    reward_scaling = args.reward_scale
    adaptive_lambda_sharpe = args.adaptive_lambda_sharpe
    adaptive_lambda_cvar = args.adaptive_lambda_cvar
    adaptive_lambda_turnover = args.adaptive_lambda_turnover
    adaptive_crisis_gain_sharpe = args.adaptive_crisis_gain_sharpe
    adaptive_crisis_gain_cvar = args.adaptive_crisis_gain_cvar
    adaptive_dsr_beta = args.adaptive_dsr_beta
    adaptive_cvar_window = args.adaptive_cvar_window
    meta_path = os.path.join(os.path.dirname(model_path), "env_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as meta_fp:
            env_meta = json.load(meta_fp)
        expected_obs_dim = env_meta.get("obs_dim")
        use_weighted_action = env_meta.get("use_weighted_action", use_weighted_action)
        weight_slippage = env_meta.get("weight_slippage", weight_slippage)
        weight_transaction_cost = env_meta.get(
            "weight_transaction_cost", weight_transaction_cost
        )
        reward_scaling = env_meta.get("reward_scaling", reward_scaling)
        adaptive_lambda_sharpe = env_meta.get(
            "adaptive_lambda_sharpe", adaptive_lambda_sharpe
        )
        adaptive_lambda_cvar = env_meta.get(
            "adaptive_lambda_cvar", adaptive_lambda_cvar
        )
        adaptive_lambda_turnover = env_meta.get(
            "adaptive_lambda_turnover", adaptive_lambda_turnover
        )
        adaptive_crisis_gain_sharpe = env_meta.get(
            "adaptive_crisis_gain_sharpe",
            env_meta.get("adaptive_crisis_gain", adaptive_crisis_gain_sharpe),
        )
        adaptive_crisis_gain_cvar = env_meta.get(
            "adaptive_crisis_gain_cvar",
            adaptive_crisis_gain_cvar,
        )
        adaptive_dsr_beta = env_meta.get("adaptive_dsr_beta", adaptive_dsr_beta)
        adaptive_cvar_window = env_meta.get(
            "adaptive_cvar_window", adaptive_cvar_window
        )

        # reward_type이 저장되어 있으면 CLI 인자보다 우선시한다.
        if env_meta.get("reward_type"):
            if args.reward_type and args.reward_type != env_meta["reward_type"]:
                print(
                    f"  Warning: overriding CLI reward_type ({args.reward_type}) with checkpoint value ({env_meta['reward_type']})"
                )
            args.reward_type = env_meta["reward_type"]
        args.adaptive_lambda_sharpe = adaptive_lambda_sharpe
        args.adaptive_lambda_cvar = adaptive_lambda_cvar
        args.adaptive_lambda_turnover = adaptive_lambda_turnover
        args.adaptive_crisis_gain_sharpe = adaptive_crisis_gain_sharpe
        args.adaptive_crisis_gain_cvar = adaptive_crisis_gain_cvar
        args.adaptive_dsr_beta = adaptive_dsr_beta
        args.adaptive_cvar_window = adaptive_cvar_window

    # evaluate.py의 evaluate_model() 재사용
    (
        portfolio_values,
        exec_returns,
        value_returns,
        irt_data,
        metrics,
        _artefacts,
    ) = evaluate_model(
        model_path=model_path,
        model_class=SAC,
        test_start=args.test_start,
        test_end=args.test_end,
        stock_tickers=DOW_30_TICKER,
        tech_indicators=INDICATORS,
        initial_amount=1000000,
        verbose=True,
        reward_type=args.reward_type,
        lambda_dsr=args.lambda_dsr,
        lambda_cvar=args.lambda_cvar,
        expected_obs_dim=expected_obs_dim,
        use_weighted_action=use_weighted_action,
        weight_slippage=weight_slippage,
        weight_transaction_cost=weight_transaction_cost,
        reward_scaling=reward_scaling,
        adaptive_lambda_sharpe=adaptive_lambda_sharpe,
        adaptive_lambda_cvar=adaptive_lambda_cvar,
        adaptive_lambda_turnover=adaptive_lambda_turnover,
        adaptive_crisis_gain_sharpe=adaptive_crisis_gain_sharpe,
        adaptive_crisis_gain_cvar=adaptive_crisis_gain_cvar,
        adaptive_dsr_beta=adaptive_dsr_beta,
        adaptive_cvar_window=adaptive_cvar_window,
    )

    # 결과 추출
    final_value = portfolio_values[-1]
    total_return = metrics["total_return"]
    step = metrics["n_steps"]

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print("\n[Period]")
    print(f"  Start: {args.test_start}")
    print(f"  End: {args.test_end}")
    print(f"  Steps: {step}")

    print("\n[Returns]")
    print(f"  Total return: {total_return*100:.2f}%")
    print(f"  Annualized return: {metrics['annualized_return']*100:.2f}%")

    print("\n[Risk Metrics]")
    print(f"  Volatility (annualized): {metrics['volatility']*100:.2f}%")
    print(f"  Maximum drawdown: {metrics['max_drawdown']*100:.2f}%")

    print("\n[Risk-Adjusted Returns]")
    print(f"  Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino ratio: {metrics['sortino_ratio']:.3f}")
    print(f"  Calmar ratio: {metrics['calmar_ratio']:.3f}")

    print("\n[Portfolio Value]")
    print("  Initial: $1,000,000.00")
    print(f"  Final: ${final_value:,.2f}")
    print(f"  Profit/Loss: ${final_value - 1000000:,.2f}")

    print("\n" + "=" * 70)

    _log_returns_to_tensorboard(os.path.dirname(model_path), metrics)

    # 6. 시각화 (기본 활성화)
    if not args.no_plot and irt_data is not None:
        from finrl.evaluation.visualizer import plot_all

        # log_dir 결정
        if model_path:
            log_dir = os.path.dirname(model_path)
        elif args.checkpoint:
            log_dir = os.path.dirname(args.checkpoint)
        else:
            log_dir = "evaluation_results"

        output_dir = args.output_plot or os.path.join(log_dir, "evaluation_plots")

        print(f"\n[Visualization]")
        print(f"  Output directory: {output_dir}")

        plot_all(
            portfolio_values=np.array(portfolio_values),
            dates=None,
            output_dir=output_dir,
            irt_data=irt_data,
        )

    # 7. JSON 저장 (기본 활성화)
    if not args.no_json and irt_data is not None:
        from finrl.evaluation.visualizer import save_evaluation_results
        from pathlib import Path

        # log_dir 결정
        if model_path:
            log_dir = os.path.dirname(model_path)
        elif args.checkpoint:
            log_dir = os.path.dirname(args.checkpoint)
        else:
            log_dir = "evaluation_results"

        # evaluation_results 구조 구축
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        results = {
            "returns": returns,
            "values": np.array(portfolio_values),
            "weights": irt_data["weights"],  # 목표가중
            "actual_weights": irt_data["actual_weights"],  # 체결 이후 실제 비중 시계열
            "crisis_levels": irt_data["crisis_levels"],
            "crisis_types": irt_data["crisis_types"],
            "prototype_weights": irt_data["prototype_weights"],
            "w_rep": irt_data["w_rep"],
            "w_ot": irt_data["w_ot"],
            "eta": irt_data["eta"],
            "alpha_c": irt_data["alpha_c"],
            "cost_matrices": irt_data["cost_matrices"],
            "symbols": irt_data["symbols"],
            "metrics": metrics,
        }

        # Config (IRT 파라미터 포함)
        config = {
            "irt": {
                "alpha": args.alpha,
                "alpha_min": args.alpha_min,
                "alpha_max": args.alpha_max if args.alpha_max else args.alpha,
                "eps": args.eps,
                "max_iters": args.max_iters,
                "replicator_temp": args.replicator_temp,
                "ema_beta": args.ema_beta,
                "eta_0": args.eta_0,
                "eta_1": args.eta_1,
                "gamma": args.gamma,
                "alpha_update_rate": args.alpha_update_rate,
                "alpha_feedback_gain": args.alpha_feedback_gain,
                "alpha_feedback_bias": args.alpha_feedback_bias,
                "directional_decay_min": args.directional_decay_min,
                "alpha_noise_std": args.alpha_noise_std,
                "alpha_crisis_source": args.alpha_crisis_source,
                # 위기 감지 가중치 및 바이어스 관련 파라미터
                "w_r": args.w_r,
                "w_s": args.w_s,
                "w_c": args.w_c,
                "eta_b": args.eta_b,
            }
        }

        print(f"\n[JSON output]")
        print(f"  Output directory: {log_dir}")
        save_evaluation_results(results, Path(log_dir), config)

    return {
        "portfolio_values": portfolio_values,
        "final_value": final_value,
        "total_return": total_return,
        "metrics": metrics,
        "steps": step,
    }


def main():
    parser = argparse.ArgumentParser(description="IRT 정책 학습/평가")

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "both", "test"],
        help="실행 모드 (기본: train; 평가만 수행하려면 'eval')",
    )

    # Data period (defaults from config.py)
    parser.add_argument(
        "--train-start",
        type=str,
        default=TRAIN_START_DATE,
        help=f"훈련 시작일 (기본: {TRAIN_START_DATE})",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default=TRAIN_END_DATE,
        help=f"훈련 종료일 (기본: {TRAIN_END_DATE})",
    )
    parser.add_argument(
        "--test-start",
        type=str,
        default=TEST_START_DATE,
        help=f"테스트 시작일 (기본: {TEST_START_DATE})",
    )
    parser.add_argument(
        "--test-end",
        type=str,
        default=TEST_END_DATE,
        help=f"테스트 종료일 (기본: {TEST_END_DATE})",
    )

    # Training settings
    parser.add_argument(
        "--episodes",
        type=int,
        default=60,
        help="학습 에피소드 수 (기본: 60)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현성을 위한 난수 시드 (기본: 42)",
    )
    parser.add_argument(
        "--output", type=str, default="logs", help="출력 디렉터리 (기본: logs)"
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,  # 초기에는 1e-2였으나 기울기 전달 강화를 위해 1.0으로 상향
        help="트레이딩 환경 내부에 적용하는 보상 스케일링 계수 (기본: 1.0)",
    )
    parser.add_argument(
        "--env-diversify",
        choices=["off", "basic", "vector"],
        default="vector",
        help="환경 다변화 전략 (기본: vector)",
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="고급 설정 옵션 표시 후 종료",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="env-diversify=vector일 때 병렬 환경 수 (기본: 자동)",
    )
    parser.add_argument(
        "--start-offset-mode",
        choices=["random", "uniform", "rolling"],
        default="random",
        help="다변화 학습용 오프셋 샘플링 모드 (기본: random)",
    )
    parser.add_argument(
        "--window-step",
        type=int,
        default=252,
        help="롤링 오프셋 사용 시 윈도우 스텝(일) (기본: 252)",
    )
    parser.add_argument(
        "--domain-rand-tx",
        type=float,
        default=0.25,
        help="거래비용 스케일링에 대한 균등 도메인 랜덤화 크기 (기본: 0.25)",
    )
    parser.add_argument(
        "--domain-rand-slippage",
        type=float,
        default=0.25,
        help="슬리피지 스케일링에 대한 균등 도메인 랜덤화 크기 (기본: 0.25)",
    )

    # IRT parameters
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=128,
        help="IRT 임베딩 차원 (기본: 128)",
    )
    parser.add_argument(
        "--m-tokens", type=int, default=6, help="에피토프 토큰 수 (기본: 6)"
    )
    parser.add_argument(
        "--M-proto", type=int, default=8, help="프로토타입 수 (기본: 8)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="OT-Replicator 혼합 비율의 기본값 (기본: 0.45)",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=0.05,
        help="위기 상황에서 허용되는 α 최소값 (기본: 0.05)",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=0.75,  # OT 기여를 넓게 허용하기 위해 0.75로 설정
        help="평시 α 상한 (IRT의 유연성을 확보하기 위해 확장)",
    )
    parser.add_argument(
        "--alpha-update-rate",
        type=float,
        default=0.85,
        help="alpha_c 상태 적응 업데이트율 (기본: 0.85)",
    )
    parser.add_argument(
        "--alpha-feedback-gain",
        type=float,
        default=0.25,
        help="Sharpe 피드백 이득 (기본: 0.25)",
    )
    parser.add_argument(
        "--alpha-feedback-bias",
        type=float,
        default=0.0,
        help="Sharpe 피드백 바이어스 (기본: 0.0)",
    )
    parser.add_argument(
        "--directional-decay-min",
        type=float,
        default=0.05,
        help="α 경계 부근 최소 방향성 감쇠 계수 (기본: 0.05)",
    )
    parser.add_argument(
        "--alpha-noise-std",
        type=float,
        default=0.02,
        help="학습 중 α에 주입하는 가우시안 노이즈 표준편차 (기본: 0.02)",
    )
    parser.add_argument(
        "--alpha-crisis-source",
        type=str,
        default="pre_guard",
        choices=["pre_guard", "post_guard"],
        help="alpha_c에 사용할 위기 신호 소스 (기본: pre_guard)",
    )
    parser.add_argument(
        "--ema-beta",
        type=float,
        default=0.50,
        help="프로토타입 가중치 EMA 계수 (기본: 0.50)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.05,
        help="Sinkhorn 반복의 엔트로피 계수 (기본: 0.05)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=30,
        help="Sinkhorn 최대 반복 횟수 (기본: 30)",
    )
    parser.add_argument(
        "--replicator-temp",
        type=float,
        default=0.4,
        help="Replicator 소프트맥스 온도 (기본: 0.4, 선택 강화)",
    )
    parser.add_argument(
        "--eta-0",
        type=float,
        default=0.05,
        help="Replicator 기본 학습률 (기본: 0.05)",
    )
    parser.add_argument(
        "--eta-1",
        type=float,
        default=0.12,
        help="위기 레벨에 비례하는 Replicator 학습률 증가량 (기본: 0.12)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.90,
        help="OT 비용 함수에서 공자극 내적에 곱해지는 가중치 (기본: 0.90)",
    )
    parser.add_argument(
        "--market-feature-dim",
        type=int,
        default=12,
        help="T-Cell 입력으로 사용할 시장 특성 차원 (기본: 12)",
    )
    # Dirichlet 및 행동 온도 관련 하이퍼파라미터
    parser.add_argument(
        "--dirichlet-min",
        type=float,
        default=0.05,
        help="Dirichlet 집중도 하한 (기본: 0.05, 선택적 집중을 강화)",
    )
    parser.add_argument(
        "--dirichlet-max",
        type=float,
        default=6.0,
        help="Dirichlet 집중도 상한 (프로토타입을 더 뚜렷하게 만듦)",
    )
    parser.add_argument(
        "--action-temp",
        type=float,
        default=0.50,
        help="행동 소프트맥스 온도 (기본: 0.50, 레짐 전환을 부드럽게 함)",
    )
    parser.add_argument(
        "--action-temp-final",
        type=float,
        default=None,
        help="행동 온도의 선형 감쇠 목표값 (기본: 사용하지 않음)",
    )
    parser.add_argument(
        "--action-temp-anneal-steps",
        type=int,
        default=0,
        help="행동 온도를 감쇠시키는 스텝 수 (기본: 0, 즉시 적용)",
    )
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=None,
        help="SAC 기본 타깃 엔트로피를 수동으로 지정 (기본: 자동 계산)",
    )
    parser.add_argument(
        "--target-entropy-mult",
        type=float,
        default=None,
        help="SAC 타깃 엔트로피에 곱할 배율 (기본: 사용하지 않음)",
    )
    parser.add_argument(
        "--utd-ratio",
        type=float,
        default=None,
        help="SAC의 update-to-data 비율 (기본: 알고리즘 내부 휴리스틱)",
    )

    # DSR + CVaR 보상 및 위험 민감 설정
    parser.add_argument(
        "--reward-type",
        type=str,
        default="adaptive_risk",
        choices=["basic", "dsr_cvar", "adaptive_risk"],
        help="사용할 보상 함수 유형 (기본: adaptive_risk)",
    )
    parser.add_argument(
        "--lambda-dsr",
        type=float,
        default=0.15,
        help="DSR 보상 항 가중치 (기본: 0.15)",
    )
    parser.add_argument(
        "--lambda-cvar",
        type=float,
        default=0.05,
        help="CVaR 패널티 가중치 (기본: 0.05)",
    )
    parser.add_argument(
        "--adaptive-lambda-sharpe",
        type=float,
        default=0.20,
        help="적응형 보상에서 Sharpe 항 가중치 λ_S (기본: 0.20)",
    )
    parser.add_argument(
        "--adaptive-lambda-cvar",
        type=float,
        default=0.40,
        help="적응형 보상에서 CVaR 패널티 가중치 β (기본: 0.40)",
    )
    parser.add_argument(
        "--adaptive-lambda-turnover",
        type=float,
        default=0.0,
        help="회전율 패널티 μ (기본: 0.0, NAV에서 이미 비용을 차감하지 않는 경우만 양수 설정)",
    )
    parser.add_argument(
        "--adaptive-crisis-gain",
        dest="adaptive_crisis_gain_sharpe",
        type=float,
        default=-0.15,
        help="위기 구간에서 Sharpe 항의 가중을 조정하는 계수 g_S (기본: -0.15)",
    )
    parser.add_argument(
        "--adaptive-crisis-gain-cvar",
        type=float,
        default=0.25,
        help="위기 구간에서 CVaR 패널티를 강화하는 계수 g_C (기본: 0.25)",
    )
    parser.add_argument(
        "--adaptive-dsr-beta",
        type=float,
        default=0.92,
        help="DSR EMA 계수 β (기본: 0.92)",
    )
    parser.add_argument(
        "--adaptive-cvar-window",
        type=int,
        default=40,
        help="CVaR 추정에 사용할 윈도우 길이 (기본: 40)",
    )

    # 위기 감지 신호 및 바이어스 보정 관련 하이퍼파라미터
    parser.add_argument(
        "--w-r",
        type=float,
        default=0.55,
        help="T-Cell 시장 신호 가중치 w_r (기본: 0.55)",
    )
    parser.add_argument(
        "--w-s",
        type=float,
        default=-0.25,
        help="Sharpe 신호 가중치 w_s (기본: -0.25)",
    )
    parser.add_argument(
        "--w-c",
        type=float,
        default=0.20,
        help="CVaR 신호 가중치 w_c (기본: 0.20)",
    )
    parser.add_argument(
        "--eta-b",
        type=float,
        default=2e-2,
        help="위기 바이어스 보정을 위한 초기 학습률 (기본: 2e-2)",
    )
    parser.add_argument(
        "--eta-b-min",
        type=float,
        default=2e-3,
        help="학습률 감쇠 이후의 최소값 (기본: 2e-3)",
    )
    parser.add_argument(
        "--eta-b-decay-steps",
        type=int,
        default=30000,
        help="바이어스 학습률 코사인 감쇠 구간 길이 (기본: 30000)",
    )
    parser.add_argument(
        "--eta-b-warmup-steps",
        type=int,
        default=10000,
        help="바이어스 학습률 워밍업 기간 (기본: 10000)",
    )
    parser.add_argument(
        "--eta-b-warmup-value",
        type=float,
        default=0.05,
        help="워밍업 단계에서 사용할 바이어스 학습률 값 (기본: 0.05)",
    )
    parser.add_argument(
        "--eta-T",
        dest="eta_T",
        type=float,
        default=1e-2,
        help="위기 온도 보정 학습률 (기본: 0.01)",
    )
    parser.add_argument(
        "--p-star",
        type=float,
        default=0.35,
        help="위기 레짐 목표 비중 p* (기본: 0.35)",
    )
    parser.add_argument(
        "--temperature-min",
        type=float,
        default=0.9,
        help="적응형 온도 하한 (기본: 0.9)",
    )
    parser.add_argument(
        "--temperature-max",
        type=float,
        default=1.2,
        help="적응형 온도 상한 (기본: 1.2)",
    )
    parser.add_argument(
        "--stat-momentum",
        type=float,
        default=0.92,
        help="위기 신호 통계를 위한 EMA 모멘텀 (기본: 0.92)",
    )
    parser.add_argument(
        "--crisis-guard-rate-init",
        type=float,
        default=0.07,
        help="워밍업 기간 동안 사용할 위기 가드 비율 초기값 (기본: 0.07)",
    )
    parser.add_argument(
        "--crisis-guard-rate-final",
        type=float,
        default=0.02,
        help="워밍업 이후 적용되는 위기 가드 비율 (기본: 0.02)",
    )
    parser.add_argument(
        "--crisis-guard-warmup-steps",
        type=int,
        default=7500,
        help="위기 가드 비율 스케줄 워밍업 스텝 수 (기본: 7500)",
    )
    parser.add_argument(
        "--hysteresis-up",
        type=float,
        default=0.52,
        help="이전 레짐이 정상일 때 위기 레짐으로 전환하는 임계값 (기본: 0.52)",
    )
    parser.add_argument(
        "--hysteresis-down",
        type=float,
        default=0.42,
        help="이전 레짐이 위기일 때 정상 레짐으로 복귀하는 임계값 (기본: 0.42)",
    )
    parser.add_argument(
        "--hysteresis-quantile",
        type=float,
        default=0.72,
        help="적응형 히스테리시스 임계값을 계산할 때 사용하는 분위수 (기본: 0.72)",
    )
    parser.add_argument(
        "--k-s",
        dest="k_s",
        type=float,
        default=6.0,
        help="Sharpe z-score를 시그모이드로 변환할 때 사용하는 기울기 (기본: 6.0)",
    )
    parser.add_argument(
        "--k-c",
        dest="k_c",
        type=float,
        default=6.0,
        help="CVaR z-score 시그모이드 변환 기울기 (기본: 6.0)",
    )
    parser.add_argument(
        "--k-b",
        dest="k_b",
        type=float,
        default=4.0,
        help="기본 위기 확률 시그모이드 변환 기울기 (기본: 4.0)",
    )
    parser.add_argument(
        "--crisis-guard-rate",
        type=float,
        default=None,
        help="위기 가드 비율을 고정값으로 강제하고 싶을 때 사용하는 옵션",
    )

    # Evaluation only
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="평가용 체크포인트 경로 (--mode test에 필수)",
    )

    # Visualization (기본값: 모두 저장)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="평가 플롯 저장 비활성화 (기본: 활성화)",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=None,
        help="플롯 출력 디렉터리 (기본: log_dir/evaluation_plots)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="평가 결과 JSON 저장 비활성화 (기본: 활성화)",
    )

    base_visible = {
        "--mode",
        "--reward-type",
        "--episodes",
        "--env-diversify",
        "--advanced",
    }
    advanced_descriptions: List[Tuple[str, str]] = []
    for action in list(parser._actions):
        option_strings = tuple(action.option_strings)
        if not option_strings:
            continue
        if any(opt in ("-h", "--help") for opt in option_strings):
            continue
        if any(opt in base_visible for opt in option_strings):
            continue
        if action.help not in (None, argparse.SUPPRESS):
            label = ", ".join(option_strings)
            advanced_descriptions.append((label, action.help))
        action.help = argparse.SUPPRESS

    args = parser.parse_args()

    if getattr(args, "advanced", False):
        print("Advanced options:\n")
        for label, description in sorted(
            advanced_descriptions, key=lambda item: item[0]
        ):
            print(f"  {label}\n    {description}")
        return

    if args.mode == "test":
        args.mode = "eval"

    if args.env_diversify == "vector":
        if args.n_envs is None:
            args.n_envs = 8
    elif args.n_envs is None:
        args.n_envs = 1

    banner = (
        f"reward={args.reward_type} | "
        f"episodes={args.episodes} | "
        f"diversify={args.env_diversify} | "
        f"n_envs={args.n_envs}"
    )
    print(f"[Run] {banner}")

    # Execute
    if args.mode == "train":
        log_dir, model_path = train_irt(args)
        print(f"\nTraining complete. Results directory: {log_dir}")

    elif args.mode == "eval":
        results = test_irt(args)
        print(
            f"\nEvaluation complete. Final return: {results['total_return']*100:.2f}%"
        )

    elif args.mode == "both":
        log_dir, model_path = train_irt(args)
        print(
            "\nTraining finished. Starting evaluation with the same configuration...\n"
        )
        results = test_irt(args, model_path=model_path)
        print("\nFull pipeline completed.")
        print(f"  Training outputs: {log_dir}")
        print(f"  Final return: {results['total_return']*100:.2f}%")


if __name__ == "__main__":
    main()
