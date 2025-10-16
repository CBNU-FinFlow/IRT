# scripts/evaluate.py
# 학습된 모델을 다양한 방식으로 평가하고 시각화 결과를 생성하는 유틸리티를 제공한다.

"""
모델 상세 평가 및 시각화 스크립트

지원하는 평가 방식:
- direct: Stable Baselines3 모델을 직접 로드하여 평가 (scripts/train.py와 연동)
- drlagent: FinRL DRLAgent의 `DRL_prediction()`을 사용한 표준 평가

사용 예:
    # Direct 방식 (기본)
    python scripts/evaluate.py --model logs/sac/20251004_120000/sac_final.zip --save-plot

    # DRLAgent 방식 (FinRL 표준)
    python scripts/evaluate.py --model trained_models/sac_50k.zip --method drlagent --save-json
"""

import argparse
import contextlib
import os
import json
import subprocess
import random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterable, Dict, Any, List

from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS, TEST_START_DATE, TEST_END_DATE
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import SAC, PPO, A2C, TD3, DDPG
from sb3_contrib.tqc import TQC


def _unwrap_env(env):
    """VecEnv 및 래퍼 체인을 모두 제거하고 실제 환경 인스턴스를 반환한다."""
    env_ref = env
    visited = set()
    while True:
        if id(env_ref) in visited:
            break
        visited.add(id(env_ref))
        if hasattr(env_ref, "envs") and getattr(env_ref, "envs"):
            env_ref = env_ref.envs[0]
            continue
        if hasattr(env_ref, "env"):
            env_ref = env_ref.env
            continue
        break
    return env_ref


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
) -> StockTradingEnv:
    """평가용 StockTradingEnv를 생성하고 보상 구성 옵션을 설정한다."""
    # 상태 공간 구성: 잔고(1) + 가격(N) + 보유량(N) + 기술지표(K*N) 구조
    # reward_type이 'dsr_cvar'이면 상태 벡터에 DSR·CVaR 특성이 추가된다.
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
        # 리스크 민감형 보상 조정 파라미터
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


def calculate_metrics(
    portfolio_values,
    initial_amount=1000000,
    weights_history=None,
    returns=None,
    executed_weights_history=None,
    turnover_target_series=None,
):
    """
    포트폴리오 가치 흐름을 기반으로 주요 수익·위험 지표를 계산한다.

    Args:
        portfolio_values: 시점별 포트폴리오 총 가치를 담은 배열
        initial_amount: 초기 투자 금액
        weights_history: 시점별 포트폴리오 비중 기록 (회전율 계산 시 사용)
        returns: 사전 계산된 수익률 (없으면 함수 내에서 계산)
        executed_weights_history: 실행된 거래 후 비중 기록
        turnover_target_series: 목표 회전율 시계열 (옵션)

    Returns:
        dict: Sharpe, Sortino, CVaR 등 평가 지표를 포함한 사전
    """
    from finrl.evaluation.metrics import (
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
        calculate_calmar_ratio,
        calculate_max_drawdown,
        calculate_var,
        calculate_cvar,
        calculate_turnover,
    )

    pv = np.asarray(portfolio_values, dtype=np.float64).reshape(-1)

    # 일별 수익률 계산 (입력이 없으면 내부에서 산출)
    if returns is None:
        if pv.size > 1:
            prev_values = np.clip(pv[:-1], 1e-8, None)
            returns = (pv[1:] - pv[:-1]) / prev_values
        else:
            returns = np.array([], dtype=np.float64)
    else:
        returns = np.asarray(returns, dtype=np.float64).reshape(-1)
        expected_len = max(pv.size - 1, 0)
        if expected_len and returns.size != expected_len:
            min_len = min(expected_len, returns.size)
            if min_len == 0:
                returns = np.array([], dtype=np.float64)
            else:
                returns = returns[-min_len:]
                pv = pv[-(min_len + 1) :]
        elif expected_len == 0:
            returns = np.array([], dtype=np.float64)

    # 총 수익률
    total_return = (pv[-1] - initial_amount) / initial_amount

    # 연환산 수익률 (거래일 252일 가정)
    n_days = returns.size
    annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

    # 연환산 변동성
    volatility = np.std(returns) * np.sqrt(252)

    # Sharpe 지수 (추가 계산 로직은 metrics.py 참조)
    sharpe_ratio = calculate_sharpe_ratio(
        returns, risk_free_rate=0.02, periods_per_year=252
    )

    # 최대 낙폭
    max_drawdown = calculate_max_drawdown(pv)

    # 칼마 지수
    calmar_ratio = calculate_calmar_ratio(returns, periods_per_year=252)

    # 소르티노 지수
    sortino_ratio = calculate_sortino_ratio(
        returns, target_return=0.02, periods_per_year=252
    )

    # VaR 및 CVaR (5% 신뢰수준)
    var_5 = calculate_var(returns, alpha=0.05)
    cvar_5 = calculate_cvar(returns, alpha=0.05)

    # 하방 표준편차
    downside_returns = returns[returns < 0]
    downside_deviation = (
        np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
    )

    # Turnover (target vs executed)
    turnover_target = 0.0
    turnover_actual = 0.0
    if turnover_target_series is not None and len(turnover_target_series) > 0:
        turnover_target = float(np.mean(turnover_target_series))
    elif weights_history is not None and len(weights_history) > 1:
        turnover_target = calculate_turnover(np.array(weights_history))
    if executed_weights_history is not None and len(executed_weights_history) > 1:
        turnover_actual = calculate_turnover(np.array(executed_weights_history))
    else:
        turnover_actual = turnover_target

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_drawdown": max_drawdown,
        "var_5": var_5,
        "cvar_5": cvar_5,
        "downside_deviation": downside_deviation,
        "avg_turnover": turnover_actual,
        "avg_turnover_target": turnover_target,
        "turnover_gap_abs": abs(turnover_actual - turnover_target),
        "final_value": pv[-1],
        "n_steps": n_days,
    }


def plot_results(
    portfolio_values,
    returns,
    output_dir,
    model_name="Model",
    model_path=None,
    irt_data=None,
    cumulative_mode: str = "log",
):
    """
    시각화 생성 (finrl.evaluation.visualizer 사용)

    일반 모델: 3개 시각화
    IRT 모델: 14개 시각화

    Args:
        portfolio_values: 포트폴리오 가치 배열
        returns: 실행 기반 수익률 배열 (decimal)
        output_dir: 출력 디렉토리
        model_name: 모델 이름
        model_path: 모델 경로 (IRT 감지용)
        irt_data: IRT 중간 데이터 (optional)
    """
    from finrl.evaluation.visualizer import plot_all

    plot_all(
        portfolio_values=np.array(portfolio_values),
        dates=None,
        output_dir=output_dir,
        irt_data=irt_data,
        returns=np.array(returns) if returns is not None else None,
        cumulative_mode=cumulative_mode,
    )


def detect_model_type(model_path):
    """모델 파일명에서 모델 타입 자동 감지"""

    filename = os.path.basename(model_path).lower()

    if "tqc" in filename:
        return "tqc", TQC
    if "sac" in filename:
        return "sac", SAC
    elif "ppo" in filename:
        return "ppo", PPO
    elif "a2c" in filename:
        return "a2c", A2C
    elif "td3" in filename:
        return "td3", TD3
    elif "ddpg" in filename:
        return "ddpg", DDPG
    else:
        # 경로에서 찾기
        path_lower = model_path.lower()
        order = [
            ("tqc", TQC),
            ("sac", SAC),
            ("ppo", PPO),
            ("a2c", A2C),
            ("td3", TD3),
            ("ddpg", DDPG),
        ]
        for name, cls in order:
            if name in path_lower:
                return name, cls

        raise ValueError(
            f"모델 타입을 자동 감지할 수 없습니다: {model_path}\n"
            f"파일명 또는 경로에 모델명(tqc/sac/ppo/a2c/td3/ddpg)을 포함하거나 --model-type 인자를 명시하세요."
        )


def evaluate_model(
    model_path,
    model_class,
    test_start,
    test_end,
    stock_tickers=DOW_30_TICKER,
    tech_indicators=INDICATORS,
    initial_amount=1000000,
    verbose=True,
    reward_type="basic",
    lambda_dsr=0.1,
    lambda_cvar=0.05,
    expected_obs_dim=None,
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
    *,
    cap_metrics: bool = False,
    xai_level: str = "off",
    xai_target: str = "critic_q",
    xai_k: int = 20,
    xai_ig_steps: int = 20,
    xai_log_interval: int = 10,
    xai_output_dir: Optional[str] = None,
):
    """
    범용 모델 평가 함수

    Args:
        model_path: 모델 파일 경로 (.zip)
        model_class: SB3 모델 클래스 (SAC, PPO 등)
        test_start: 테스트 시작 날짜 (YYYY-MM-DD)
        test_end: 테스트 종료 날짜 (YYYY-MM-DD)
        stock_tickers: 주식 티커 리스트
        tech_indicators: 기술 지표 리스트
        initial_amount: 초기 자본
        verbose: 진행 상황 출력 여부
        reward_type: 보상 함수 타입 (basic/dsr_cvar/adaptive_risk). IRT 모델은 자동 감지.
        lambda_dsr: DSR 보상 가중치
        lambda_cvar: CVaR 보상 가중치
        expected_obs_dim: 학습 시점 관측 공간 차원 (검증용)
        use_weighted_action: 가중치 기반 행동 모드 사용 여부
        weight_slippage: 가중치 슬리피지 계수
        weight_transaction_cost: 가중치 거래 비용 계수
        reward_scaling: 환경 보상 스케일링 계수
        adaptive_lambda_sharpe: Adaptive risk reward Sharpe 가중치
        adaptive_lambda_cvar: Adaptive risk reward CVaR 가중치
        adaptive_lambda_turnover: Adaptive risk reward turnover 가중치
        adaptive_crisis_gain_sharpe: Adaptive risk reward Sharpe 위기 계수
        adaptive_crisis_gain_cvar: Adaptive risk reward CVaR 위기 계수
        adaptive_dsr_beta: Adaptive risk reward DSR EMA 계수
        adaptive_cvar_window: Adaptive risk reward CVaR 추정 윈도우
        cap_metrics: 메트릭 산출 전에 수익률 클리핑 여부 (False 권장)
        xai_level: XAI 출력 수준 ('off' / 'light' / 'full')
        xai_target: 특징 기여도 대상 ('critic_q' 또는 'log_prob')
        xai_k: 특징 기여도 샘플 스텝 수(최대)
        xai_ig_steps: Integrated Gradients 분해 스텝 수 (1이면 Grad×Input)
        xai_log_interval: 프로토타입 시계열 기록 간격
        xai_output_dir: XAI 파일 출력 경로 (None이면 모델 폴더 내 xai/ 사용)

    Returns:
        portfolio_values: 포트폴리오 가치 배열
        execution_returns: 거래 비용 반영 실행 기반 수익률 (정제됨)
        value_returns: 포트폴리오 가치 기반 보조 수익률
        irt_data: IRT 중간 데이터 (IRT 모델인 경우, 아니면 None)
        metrics: 성능 지표 딕셔너리
    """
    if verbose:
        print(f"\n[평가 방식: Direct - SB3 predict()]")

    # 1. 데이터 준비
    if verbose:
        print(f"\n[1/4] 평가 데이터 다운로드 중...")
    df = YahooDownloader(
        start_date=test_start, end_date=test_end, ticker_list=stock_tickers
    ).fetch_data()
    if verbose:
        print(f"  다운로드 완료: {df.shape[0]}행")

    # 2. Feature Engineering
    if verbose:
        print(f"\n[2/4] 피처 엔지니어링 수행 중...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=tech_indicators,
        use_turbulence=False,
        user_defined_feature=False,
    )
    df_processed = fe.preprocess_data(df)
    test_df = data_split(df_processed, test_start, test_end)
    if verbose:
        print(f"  평가 샘플 수: {len(test_df)}")

    # 3. 환경 및 모델 로드
    if verbose:
        print(f"\n[3/4] 모델 및 환경 설정 중...")
    stock_dim = len(test_df.tic.unique())
    if verbose:
        print(f"  실제 사용 종목 수: {stock_dim}")

    # IRT 모델은 학습 시 사용한 보상 구성을 유지해야 하므로 메타데이터와 일치시키는 것을 우선한다.
    is_irt = "irt" in model_path.lower()
    requested_reward_type = reward_type

    xai_level_normalized = (xai_level or "off").lower()
    xai_target_normalized = (xai_target or "critic_q").lower()
    if xai_target_normalized == "both":
        # critic_q: 가치 기반 설명용, log_prob: 정책 확률 관점 → both면 두 타깃 모두 계산
        xai_targets_to_use = ["critic_q", "log_prob"]
    else:
        xai_targets_to_use = [xai_target_normalized]
    xai_enabled = xai_level_normalized != "off"
    xai_collect_features = xai_level_normalized == "full"
    if xai_enabled and not is_irt:
        if verbose:
            print(
                "  XAI 옵션이 요청되었지만 IRT 모델이 아니므로 XAI 기능을 비활성화합니다."
            )
        xai_enabled = False
        xai_collect_features = False
        xai_targets_to_use = []

    # 체크포인트에 저장된 환경 메타데이터를 먼저 로드해 보상 설정과 관측 차원을 검증한다.
    env_meta = {}
    feature_columns = None
    meta_candidates = []
    try:
        model_path_obj = Path(model_path).expanduser().resolve()
        meta_candidates.append(model_path_obj.parent / "env_meta.json")
        # Many checkpoints live in .../best_model/best_model.zip
        meta_candidates.append(model_path_obj.parent.parent / "env_meta.json")
    except Exception:
        model_path_obj = None  # type: ignore[assignment]

    for candidate in meta_candidates:
        if candidate is None:
            continue
        try:
            if candidate.exists():
                with candidate.open("r") as meta_fp:
                    env_meta = json.load(meta_fp)
                break
        except (OSError, json.JSONDecodeError):
            continue

    if not env_meta:
        raise FileNotFoundError(
            "체크포인트와 같은 디렉터리에 env_meta.json이 없습니다. 평가를 위해서는 보상 설정, "
            "행동 모드, 관측 공간 구성을 검증할 메타데이터가 필요합니다."
        )

    if env_meta:
        meta_reward_type = env_meta.get("reward_type")
        if env_meta.get("feature_columns"):
            try:
                feature_columns = list(env_meta["feature_columns"])
            except Exception:
                feature_columns = None
        if meta_reward_type:
            if (
                requested_reward_type
                and requested_reward_type != meta_reward_type
                and verbose
            ):
                print(
                    f"  요청된 reward_type '{requested_reward_type}' 대신 체크포인트 값 '{meta_reward_type}'을 사용합니다."
                )
            requested_reward_type = str(meta_reward_type)
        if expected_obs_dim is None and env_meta.get("obs_dim") is not None:
            expected_obs_dim = int(env_meta["obs_dim"])
        use_weighted_action = bool(
            env_meta.get("use_weighted_action", use_weighted_action)
        )
        weight_slippage = float(env_meta.get("weight_slippage", weight_slippage))
        weight_transaction_cost = float(
            env_meta.get("weight_transaction_cost", weight_transaction_cost)
        )
        reward_scaling = float(env_meta.get("reward_scaling", reward_scaling))
        adaptive_lambda_sharpe = float(
            env_meta.get("adaptive_lambda_sharpe", adaptive_lambda_sharpe)
        )
        adaptive_lambda_cvar = float(
            env_meta.get("adaptive_lambda_cvar", adaptive_lambda_cvar)
        )
        adaptive_lambda_turnover = float(
            env_meta.get("adaptive_lambda_turnover", adaptive_lambda_turnover)
        )
        adaptive_crisis_gain_sharpe = float(
            env_meta.get(
                "adaptive_crisis_gain_sharpe",
                env_meta.get("adaptive_crisis_gain", adaptive_crisis_gain_sharpe),
            )
        )
        adaptive_crisis_gain_cvar = float(
            env_meta.get("adaptive_crisis_gain_cvar", adaptive_crisis_gain_cvar)
        )
        adaptive_dsr_beta = float(env_meta.get("adaptive_dsr_beta", adaptive_dsr_beta))
        adaptive_cvar_window = int(
            env_meta.get("adaptive_cvar_window", adaptive_cvar_window)
        )

    # IRT 모델은 메타데이터를 기반으로 보상 타입을 자동 감지하며, 불일치시 해당 값을 우선한다.
    if is_irt:
        candidate_reward_types = []
        if requested_reward_type:
            candidate_reward_types.append(requested_reward_type)
        for candidate in ["adaptive_risk", "dsr_cvar", "basic"]:
            if candidate not in candidate_reward_types:
                candidate_reward_types.append(candidate)
    else:
        candidate_reward_types = [requested_reward_type or "basic"]

    model = None
    test_env = None
    env_reward_type = None
    load_error = None
    mismatch_reasons: list[str] = []

    for candidate_reward_type in candidate_reward_types:
        env_candidate = create_env(
            test_df,
            stock_dim,
            tech_indicators,
            reward_type=candidate_reward_type,
            lambda_dsr=lambda_dsr,
            lambda_cvar=lambda_cvar,
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
        if expected_obs_dim is not None:
            obs_dim = env_candidate.observation_space.shape[0]
            if obs_dim != expected_obs_dim:
                reason = (
                    f"reward_type={candidate_reward_type}: observation dimension mismatch"
                    f" (expected {expected_obs_dim}, got {obs_dim})"
                )
                mismatch_reasons.append(reason)
                load_error = ValueError(reason)
                if hasattr(env_candidate, "close"):
                    with contextlib.suppress(AttributeError):
                        env_candidate.close()
                continue
        try:
            model = model_class.load(model_path, env=env_candidate)
            test_env = env_candidate
            env_reward_type = candidate_reward_type
            break
        except ValueError as exc:
            # 관측 공간이 맞지 않으면 다음 후보 보상 타입을 시도한다.
            if "Observation spaces do not match" not in str(exc):
                raise
            load_error = exc
            if hasattr(env_candidate, "close"):
                with contextlib.suppress(AttributeError):
                    env_candidate.close()
    if model is None or test_env is None or env_reward_type is None:
        detail = "\n".join(mismatch_reasons) if mismatch_reasons else "<none>"
        raise (
            load_error
            if load_error is not None
            else ValueError(
                "Failed to load model with any compatible reward type."
                f"\nTried reward types: {candidate_reward_types}\nReasons:\n{detail}"
            )
        )

    if verbose:
        print(f"  모델 로드에 성공했습니다")
        if is_irt:
            if env_reward_type != requested_reward_type:
                print(
                    f"  모델에서 감지된 reward_type: {env_reward_type} "
                    f"(요청 값: {requested_reward_type})"
                )
            else:
                print(f"  reward_type: {env_reward_type}")

    base_env = _unwrap_env(test_env) if test_env is not None else None

    if verbose:
        print(
            f"  환경 설정 → reward_scaling={getattr(base_env, 'reward_scaling', None)}, "
            f"use_weighted_action={getattr(base_env, 'use_weighted_action', False)}, "
            f"slippage={getattr(base_env, 'weight_slippage', None)}, "
            f"transaction_cost={getattr(base_env, 'weight_transaction_cost', None)}"
        )

    def _apply_crisis_level(env_obj, level_value) -> None:
        """정책이 계산한 위기 레벨을 평가 환경에 동기화한다."""
        if env_obj is None:
            return
        try:
            crisis_float = float(np.array(level_value).item())
        except (TypeError, ValueError):
            return

        risk_reward = getattr(env_obj, "risk_reward", None)
        if risk_reward is not None and hasattr(risk_reward, "set_crisis_level"):
            with contextlib.suppress(TypeError, ValueError):
                risk_reward.set_crisis_level(crisis_float)

        if hasattr(env_obj, "_crisis_level"):
            env_obj._crisis_level = crisis_float
        if hasattr(env_obj, "last_crisis_level"):
            env_obj.last_crisis_level = crisis_float

    class _CrisisBridgeMonitor:
        """위기 임계값과 레짐 상태를 추적하고 환경과 동기화하는 보조 클래스"""

        def __init__(
            self,
            vec_env=None,
            base_env=None,
            default_up: float = 0.55,
            default_down: float = 0.45,
        ):
            self.vec_env = vec_env
            self.base_env = base_env
            self.hysteresis_up = float(default_up)
            self.hysteresis_down = float(default_down)
            self.prev_regime = None  # type: ignore[assignment]
            self.latest_level = None  # type: ignore[assignment]
            self.steps_crisis = 0
            self.steps_normal = 0

        @staticmethod
        def _to_float(value, fallback=None):
            try:
                import torch  # type: ignore

                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
            except Exception:
                pass
            if value is None:
                return fallback
            try:
                return float(np.asarray(value).astype(np.float64).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                return fallback

        def observe_policy(self, info_dict) -> None:
            """정책에서 전달된 통계를 수집해 환경 업데이트에 반영한다."""
            if not info_dict or "crisis_level" not in info_dict:
                self.latest_level = None
                return

            self.latest_level = self._to_float(info_dict.get("crisis_level"), None)

            hysteresis_up_val = self._to_float(info_dict.get("hysteresis_up"), None)
            hysteresis_down_val = self._to_float(info_dict.get("hysteresis_down"), None)
            if hysteresis_up_val is not None:
                self.hysteresis_up = hysteresis_up_val
            if hysteresis_down_val is not None:
                self.hysteresis_down = hysteresis_down_val

        def inject_env(self) -> None:
            """가장 최근 위기 레벨을 환경으로 전달한다."""
            if self.latest_level is None:
                return
            env_target = self.base_env if self.base_env is not None else self.vec_env
            _apply_crisis_level(env_target, self.latest_level)

        def classify(self):
            """히스테리시스 규칙에 따라 레짐 레이블을 계산하고 카운터를 업데이트한다."""
            if self.latest_level is None:
                return None

            if self.prev_regime is None:
                regime = 1 if self.latest_level > 0.5 else 0
            elif self.prev_regime == 0:
                regime = 1 if self.latest_level > self.hysteresis_up else 0
            else:
                regime = 0 if self.latest_level < self.hysteresis_down else 1

            self.prev_regime = regime
            if regime == 1:
                self.steps_crisis += 1
            else:
                self.steps_normal += 1
            return regime

        def stats(self):
            return {
                "steps_crisis": self.steps_crisis,
                "steps_normal": self.steps_normal,
            }

    # 4. 평가 실행
    if verbose:
        print(f"\n[4/4] 평가 시뮬레이션 실행 중...")
    obs, _ = test_env.reset()
    if is_irt and env_reward_type == "adaptive_risk":
        target_env = base_env if base_env is not None else test_env
        _apply_crisis_level(target_env, getattr(target_env, "last_crisis_level", 0.5))
    done = False
    portfolio_values = [initial_amount]
    value_returns = []
    execution_returns = []
    transaction_costs = []
    turnover_executed = []
    turnover_target_series = []
    holdings_records: List[Dict[str, float]] = []
    trade_records: List[Dict[str, Any]] = []
    date_records: List[Optional[str]] = []
    cash_ratio_series: List[float] = []

    prev_prices = None
    if stock_dim > 0 and isinstance(obs, (np.ndarray, list, tuple)):
        obs_array = np.asarray(obs, dtype=np.float64).reshape(-1)
        if obs_array.size >= stock_dim + 1:
            prev_prices = obs_array[1 : stock_dim + 1]

    initial_state = np.asarray(
        getattr(test_env, "state", np.zeros(1)), dtype=np.float64
    )
    if stock_dim > 0 and initial_state.size >= stock_dim * 2 + 1:
        prev_holdings = initial_state[stock_dim + 1 : 2 * stock_dim + 1].astype(
            np.float64, copy=True
        )
    else:
        prev_holdings = np.zeros(stock_dim, dtype=np.float64)

    # IRT 모델 감지 (이미 위에서 설정됨, 중복 제거)

    # IRT 데이터 수집 준비
    if is_irt:
        irt_data_list = {
            "w": [],
            "w_rep": [],
            "w_ot": [],
            "crisis_levels": [],
            "crisis_levels_pre_guard": [],
            "crisis_raw": [],
            "crisis_bias": [],
            "crisis_temperature": [],
            "crisis_guard_rate": [],
            "crisis_types": [],
            "cost_matrices": [],
            "weights": [],
            "actual_weights": [],  # 체결 후 실제 비중 기록
            "eta": [],
            "alpha_c": [],
            "alpha_c_raw": [],  # 클램프 적용 전 α 값 기록
            "alpha_c_prev": [],  # 직전 스텝의 α 값 기록
            "alpha_c_decay_factor": [],
            "alpha_crisis_input": [],
            "hysteresis_up": [],
            "hysteresis_down": [],
            "turnover_target": [],
            "top_snapshots": [],
            "crisis_regime": [],  # 히스테리시스 기반 이진 레짐 결과 (0/1)
            "reward_components": {},
            "reward_components_scaled": {},
        }
        if env_reward_type == "adaptive_risk":
            irt_data_list["env_crisis_levels"] = []
            irt_data_list["env_kappa"] = []
            irt_data_list["env_delta_sharpe"] = []
    else:
        irt_data_list = {}

    bridge_monitor = (
        _CrisisBridgeMonitor(vec_env=test_env, base_env=base_env) if is_irt else None
    )

    step = 0
    xai_proto_records: List[Dict[str, Any]] = []
    xai_cash_series: List[Dict[str, float]] = []
    xai_feature_samples: List[Dict[str, Any]] = []
    xai_feature_rows: List[Dict[str, Any]] = []
    feature_sample_total = 0
    proto_top_k_default = 5
    reservoir_rng = random.Random(0)

    while not done:
        obs_before = np.asarray(obs, dtype=np.float32).copy()
        action, _ = model.predict(obs, deterministic=True)

        info_dict = None
        proto_snapshot = None
        # IRT info 수집
        if is_irt and hasattr(model.policy, "get_irt_info"):
            info_dict = model.policy.get_irt_info()
            if info_dict is not None:
                if bridge_monitor is not None:
                    bridge_monitor.observe_policy(info_dict)
                # Batch=1이므로 [0] 인덱스로 추출
                irt_data_list["w"].append(info_dict["w"][0].cpu().numpy())
                irt_data_list["w_rep"].append(info_dict["w_rep"][0].cpu().numpy())
                irt_data_list["w_ot"].append(info_dict["w_ot"][0].cpu().numpy())
                irt_data_list["crisis_levels"].append(
                    info_dict["crisis_level"][0].cpu().numpy()
                )
                if "crisis_level_pre_guard" in info_dict:
                    irt_data_list["crisis_levels_pre_guard"].append(
                        info_dict["crisis_level_pre_guard"][0].cpu().numpy()
                    )
                if "crisis_raw" in info_dict:
                    irt_data_list["crisis_raw"].append(
                        info_dict["crisis_raw"][0].cpu().numpy()
                    )
                if "crisis_bias" in info_dict:
                    irt_data_list["crisis_bias"].append(
                        info_dict["crisis_bias"].cpu().numpy()
                    )
                if "crisis_temperature" in info_dict:
                    irt_data_list["crisis_temperature"].append(
                        info_dict["crisis_temperature"].cpu().numpy()
                    )
                if "crisis_guard_rate" in info_dict:
                    irt_data_list["crisis_guard_rate"].append(
                        info_dict["crisis_guard_rate"].cpu().numpy()
                    )
                irt_data_list["crisis_types"].append(
                    info_dict["crisis_types"][0].cpu().numpy()
                )
                irt_data_list["cost_matrices"].append(
                    info_dict["cost_matrix"][0].cpu().numpy()
                )
                irt_data_list["eta"].append(info_dict["eta"][0].cpu().numpy())
                irt_data_list["alpha_c"].append(info_dict["alpha_c"][0].cpu().numpy())

                # α 관련 상세 정보를 함께 저장한다.
                if "alpha_c_raw" in info_dict:
                    irt_data_list["alpha_c_raw"].append(
                        info_dict["alpha_c_raw"][0].cpu().numpy()
                    )
                if "alpha_c_prev" in info_dict:
                    irt_data_list["alpha_c_prev"].append(
                        info_dict["alpha_c_prev"][0].cpu().numpy()
                    )
                if "alpha_c_decay_factor" in info_dict:
                    irt_data_list["alpha_c_decay_factor"].append(
                        info_dict["alpha_c_decay_factor"][0].cpu().numpy()
                    )
                if "alpha_crisis_input" in info_dict:
                    irt_data_list["alpha_crisis_input"].append(
                        info_dict["alpha_crisis_input"][0].cpu().numpy()
                    )

                # 히스테리시스 임계값을 저장해 추후 분석에 활용한다.
                hysteresis_up_val = 0.55  # 기본값
                hysteresis_down_val = 0.45  # 기본값
                if "hysteresis_up" in info_dict:
                    hysteresis_up_val = float(
                        np.asarray(info_dict["hysteresis_up"], dtype=np.float64).item()
                    )
                    irt_data_list["hysteresis_up"].append(hysteresis_up_val)
                if "hysteresis_down" in info_dict:
                    hysteresis_down_val = float(
                        np.asarray(
                            info_dict["hysteresis_down"], dtype=np.float64
                        ).item()
                    )
                    irt_data_list["hysteresis_down"].append(hysteresis_down_val)
                # Action을 weight로 변환 (simplex 정규화)
                weights = action / (action.sum() + 1e-8)
                irt_data_list["weights"].append(weights.copy())
                if xai_enabled and (step % max(1, xai_log_interval) == 0):
                    proto_weights = info_dict["w"][0].detach().cpu().numpy()
                    proto_weights = np.asarray(proto_weights, dtype=np.float64).reshape(
                        -1
                    )
                    proto_sum = np.sum(proto_weights)
                    if proto_sum <= 0:
                        proto_weights = np.ones_like(proto_weights) / max(
                            proto_weights.size, 1
                        )
                    else:
                        proto_weights = proto_weights / proto_sum
                    proto_entropy_val = float(
                        -(proto_weights * np.log(proto_weights + 1e-12)).sum()
                    )
                    alpha_vals = info_dict.get("alpha_c")
                    if alpha_vals is not None:
                        alpha_arr = alpha_vals[0].detach().cpu().numpy().reshape(-1)
                        alpha_mean_val = float(np.mean(alpha_arr))
                        alpha_std_val = float(np.std(alpha_arr))
                    else:
                        alpha_mean_val = float("nan")
                        alpha_std_val = float("nan")
                    crisis_tensor = info_dict.get("crisis_level")
                    if crisis_tensor is not None:
                        crisis_level_val = float(crisis_tensor[0].detach().cpu().item())
                    else:
                        crisis_level_val = 0.0
                    top_k = min(proto_top_k_default, proto_weights.size)
                    top_indices = np.argsort(proto_weights)[::-1][:top_k]
                    top_weights = proto_weights[top_indices]
                    action_temp_val = getattr(model.policy, "action_temp", None)
                    proto_snapshot = {
                        "step": step,
                        "proto_entropy": proto_entropy_val,
                        "alpha_c_mean": alpha_mean_val,
                        "alpha_c_std": alpha_std_val,
                        "crisis_level": crisis_level_val,
                        "action_temp": (
                            float(action_temp_val)
                            if action_temp_val is not None
                            else float("nan")
                        ),
                        "top_indices": top_indices.tolist(),
                        "top_weights": top_weights.tolist(),
                    }

        if bridge_monitor is not None and info_dict is None:
            bridge_monitor.observe_policy(None)

        next_obs, reward, done_step, truncated, info = test_env.step(action)
        done = done_step or truncated

        # 정책이 제공한 위기 정보를 환경에 반영하고 히스테리시스 기반 레짐을 분류한다.
        if bridge_monitor is not None:
            if env_reward_type == "adaptive_risk":
                bridge_monitor.inject_env()
            regime = bridge_monitor.classify()
            if regime is not None:
                irt_data_list["crisis_regime"].append(int(regime))

        # Portfolio value 계산
        state = np.asarray(test_env.state, dtype=np.float64)
        cash = float(state[0])
        prices = np.asarray(state[1 : stock_dim + 1], dtype=np.float64)
        holdings = np.asarray(
            state[stock_dim + 1 : 2 * stock_dim + 1], dtype=np.float64
        )
        prev_value = float(portfolio_values[-1])
        pv = float(cash + np.dot(prices, holdings))
        portfolio_values.append(pv)

        denom = max(prev_value, 1e-8)
        value_return = (pv - prev_value) / denom
        value_returns.append(value_return)

        total_equity = max(pv, 1e-8)
        weights_actual = (
            (prices * holdings) / total_equity if stock_dim > 0 else np.array([])
        )
        cash_ratio_val = float(max(cash, 0.0) / total_equity)

        current_date = None
        if hasattr(test_env, "date_memory") and test_env.date_memory:
            current_date = test_env.date_memory[-1]
        date_records.append(str(current_date) if current_date is not None else None)

        holdings_record: Dict[str, float] = {"step": int(step)}
        for idx in range(stock_dim):
            ticker = stock_tickers[idx] if idx < len(stock_tickers) else f"asset_{idx}"
            weight_val = (
                float(weights_actual[idx]) if idx < weights_actual.size else 0.0
            )
            holdings_record[ticker] = weight_val
        holdings_record["CASH"] = cash_ratio_val
        holdings_records.append(holdings_record)
        cash_ratio_series.append(cash_ratio_val)

        delta_positions = holdings - prev_holdings
        trade_step_records: List[Dict[str, Any]] = []
        total_trade_value = 0.0
        for idx, delta in enumerate(delta_positions):
            if abs(delta) <= 1e-8:
                continue
            ticker = stock_tickers[idx] if idx < len(stock_tickers) else f"asset_{idx}"
            price_val = float(prices[idx]) if idx < prices.size else 0.0
            trade_value = abs(delta) * price_val
            total_trade_value += trade_value
            trade_step_records.append(
                {
                    "step": int(step),
                    "timestamp": str(current_date) if current_date is not None else "",
                    "ticker": ticker,
                    "qty": float(delta),
                    "price": price_val,
                    "side": "buy" if delta > 0 else "sell",
                }
            )
        prev_holdings = holdings.copy()

        executed_weights = info.get("executed_weights")
        if executed_weights is not None:
            executed_weights = np.asarray(executed_weights, dtype=np.float64).reshape(
                -1
            )

        turnover_value = float(info.get("turnover_target", 0.0) or 0.0)
        turnover_target_series.append(turnover_value)
        if is_irt:
            irt_data_list["turnover_target"].append(turnover_value)
            reward_components = info.get("reward_components")
            if reward_components:
                scaling_factor = float(getattr(test_env, "reward_scaling", 1.0))
                for key, value in reward_components.items():
                    component_value = float(value)
                    bucket = irt_data_list["reward_components"].setdefault(key, [])
                    bucket.append(component_value)
                    bucket_scaled = irt_data_list[
                        "reward_components_scaled"
                    ].setdefault(key, [])
                    bucket_scaled.append(component_value * scaling_factor)

        # 체결 이후 실제 포트폴리오 비중을 계산해 기록한다.
        actual_weights = None
        if is_irt:
            # w^{exec}_t = (p_t ⊙ h_t) / (cash_t + Σ p_{t,i} h_{t,i} + ε)
            total_equity = pv + 1e-8
            actual_weights = (prices * holdings) / total_equity
            irt_data_list["actual_weights"].append(actual_weights)
            if env_reward_type == "adaptive_risk":
                irt_data_list["env_crisis_levels"].append(test_env.last_crisis_level)
                irt_data_list["env_kappa"].append(test_env.last_kappa)
                irt_data_list["env_delta_sharpe"].append(test_env.last_delta_sharpe)

            if len(irt_data_list["top_snapshots"]) < 50 and actual_weights is not None:
                sorted_idx = np.argsort(actual_weights)[::-1]
                top_k = int(min(5, sorted_idx.size))
                top_symbols = [stock_tickers[i] for i in sorted_idx[:top_k]]
                top_exec_weights = actual_weights[sorted_idx[:top_k]].tolist()
                target_slice = None
                if irt_data_list["weights"]:
                    last_target = np.asarray(irt_data_list["weights"][-1])
                    target_slice = last_target[sorted_idx[:top_k]].tolist()
                current_date = None
                if hasattr(test_env, "date_memory") and test_env.date_memory:
                    current_date = test_env.date_memory[-1]
                snapshot = {
                    "step": step,
                    "date": str(current_date) if current_date is not None else None,
                    "tickers": top_symbols,
                    "weights": top_exec_weights,
                    "target_weights": target_slice,
                }
                irt_data_list["top_snapshots"].append(snapshot)

        crisis_for_step = (
            proto_snapshot["crisis_level"]
            if proto_snapshot is not None
            else float(info.get("crisis_level", 0.0))
        )
        if xai_enabled:
            cash_weight_val = info.get("cash_weight")
            if cash_weight_val is not None:
                xai_cash_series.append(
                    {
                        "step": step,
                        "cash_weight": float(cash_weight_val),
                        "crisis_level": float(crisis_for_step),
                    }
                )
        if proto_snapshot is not None:
            proto_snapshot["cash_weight"] = float(info.get("cash_weight", float("nan")))
            proto_snapshot["sum_weights"] = float(info.get("sum_weights", float("nan")))
            proto_snapshot["kappa_sharpe"] = float(
                info.get("kappa_sharpe", float("nan"))
            )
            proto_snapshot["kappa_cvar"] = float(info.get("kappa_cvar", float("nan")))
            proto_snapshot["turnover_executed"] = float(
                info.get("turnover_executed", float("nan"))
            )
            xai_proto_records.append(proto_snapshot)
        if xai_collect_features and info_dict is not None:
            feature_sample_total += 1
            sample_payload = {
                "step": step,
                "obs": obs_before.copy(),
                "action": np.asarray(action, dtype=np.float32).copy(),
                "crisis_level": float(crisis_for_step),
            }
            if len(xai_feature_samples) < max(1, xai_k):
                xai_feature_samples.append(sample_payload)
            else:
                j = reservoir_rng.randint(0, feature_sample_total - 1)
                if j < max(1, xai_k):
                    xai_feature_samples[j] = sample_payload

        tc_value = float(info.get("transaction_cost", 0.0) or 0.0)
        transaction_costs.append(tc_value)
        if trade_step_records:
            for rec in trade_step_records:
                if total_trade_value > 0.0 and tc_value > 0.0:
                    share = (
                        abs(rec["qty"]) * rec["price"] / total_trade_value
                        if rec["price"] > 0
                        else 0.0
                    )
                    rec["tx_cost"] = float(tc_value * share)
                else:
                    rec["tx_cost"] = 0.0
                trade_records.append(rec)

        current_prices = None
        if stock_dim > 0 and isinstance(next_obs, (np.ndarray, list, tuple)):
            next_obs_array = np.asarray(next_obs, dtype=np.float64).reshape(-1)
            if next_obs_array.size >= stock_dim + 1:
                current_prices = next_obs_array[1 : stock_dim + 1]

        exec_return = value_return
        if prev_prices is not None and current_prices is not None:
            price_denominator = np.clip(prev_prices, 1e-8, None)
            price_relatives = (current_prices - price_denominator) / price_denominator
            weight_vector = executed_weights
            if weight_vector is None and actual_weights is not None:
                weight_vector = actual_weights
            if weight_vector is not None and weight_vector.size == price_relatives.size:
                exec_return = float(
                    np.dot(weight_vector, price_relatives) - tc_value / denom
                )
        execution_returns.append(exec_return)

        if hasattr(test_env, "_last_turnover_executed"):
            turnover_executed.append(
                float(getattr(test_env, "_last_turnover_executed"))
            )
        else:
            turnover_executed.append(0.0)

        obs = next_obs
        prev_prices = current_prices
        step += 1

    if verbose:
        print(f"  Evaluation completed: {step} steps")

    if bridge_monitor is not None:
        stats = bridge_monitor.stats()
        total_reg = stats.get("steps_crisis", 0) + stats.get("steps_normal", 0)
        expected_reg = len(irt_data_list.get("crisis_levels", []))
        if total_reg == 0:
            raise RuntimeError(
                "Crisis regime classification produced zero samples. Verify bridge wiring."
            )
        if expected_reg and total_reg != expected_reg:
            print(
                f"⚠️  Crisis regime length mismatch (classified {total_reg} vs {expected_reg}).",
            )

    # 수익률 시리즈 (raw + sanitize)
    from finrl.evaluation.visualizer import sanitize_returns

    execution_returns_raw = np.array(execution_returns, dtype=np.float64)
    value_returns_raw = np.array(value_returns, dtype=np.float64)
    execution_returns_array = sanitize_returns(execution_returns_raw)
    value_returns_array = sanitize_returns(value_returns_raw)
    transaction_costs_array = np.array(transaction_costs, dtype=np.float64)
    turnover_executed_array = np.array(turnover_executed, dtype=np.float64)
    turnover_target_array = np.array(turnover_target_series, dtype=np.float64)
    if (
        turnover_target_array.size
        and turnover_target_array.size != turnover_executed_array.size
    ):
        raise RuntimeError(
            "Turnover series length mismatch between target and executed. "
            f"target={turnover_target_array.size}, executed={turnover_executed_array.size}"
        )

    if date_records and len(date_records) > execution_returns_array.size:
        date_records = date_records[: execution_returns_array.size]
    if holdings_records and len(holdings_records) > execution_returns_array.size:
        holdings_records = holdings_records[: execution_returns_array.size]

    # IRT 데이터 변환
    irt_data = None
    if is_irt and irt_data_list["w"]:
        irt_data = {
            "w_rep": np.array(irt_data_list["w_rep"]),  # [T, M]
            "w_ot": np.array(irt_data_list["w_ot"]),  # [T, M]
            "weights": np.array(irt_data_list["weights"]),  # [T, N] 목표 비중
            "actual_weights": np.array(
                irt_data_list["actual_weights"]
            ),  # [T, N] 체결 후 실제 비중
            "crisis_levels": np.array(irt_data_list["crisis_levels"]).squeeze(),  # [T]
            "crisis_levels_pre_guard": (
                np.array(irt_data_list["crisis_levels_pre_guard"]).squeeze()
                if irt_data_list["crisis_levels_pre_guard"]
                else np.array([], dtype=np.float64)
            ),
            "crisis_regime": np.array(
                irt_data_list["crisis_regime"], dtype=np.int32
            ),  # [T] 히스테리시스 기반 이진 분류 결과
            "crisis_types": np.array(irt_data_list["crisis_types"]),  # [T, K]
            "prototype_weights": np.array(irt_data_list["w"]),  # [T, M]
            "cost_matrices": np.array(irt_data_list["cost_matrices"]),  # [T, m, M]
            "eta": np.array(irt_data_list["eta"]).squeeze(),  # [T]
            "alpha_c": np.array(irt_data_list["alpha_c"]).squeeze(),  # [T]
            "alpha_crisis_input": (
                np.array(irt_data_list["alpha_crisis_input"]).squeeze()
                if irt_data_list["alpha_crisis_input"]
                else np.array([], dtype=np.float64)
            ),
            "hysteresis_up": np.array(irt_data_list["hysteresis_up"], dtype=np.float64),
            "hysteresis_down": np.array(
                irt_data_list["hysteresis_down"], dtype=np.float64
            ),
            "symbols": stock_tickers[:stock_dim],  # 실제 주식 수만큼
            "metrics": None,  # 호출자가 calculate_metrics()로 계산
            "returns": execution_returns_array,
            "returns_exec": execution_returns_array,
            "returns_value": value_returns_array,
            "transaction_costs": transaction_costs_array,
            "turnover_executed": turnover_executed_array,
            "turnover_target": np.array(
                irt_data_list["turnover_target"], dtype=np.float64
            ),
            "returns_raw": execution_returns_raw,
            "returns_sanitized": execution_returns_array,
            "value_returns_raw": value_returns_raw,
            "value_returns_sanitized": value_returns_array,
            "reward_components": {
                key: np.array(values, dtype=np.float64)
                for key, values in irt_data_list["reward_components"].items()
            },
            "reward_components_scaled": {
                key: np.array(values, dtype=np.float64)
                for key, values in irt_data_list["reward_components_scaled"].items()
            },
            "top_snapshots": irt_data_list["top_snapshots"],
            "dates": list(date_records),
            "holdings_timeseries": holdings_records.copy(),
            "trades": trade_records.copy(),
        }
        # α 관련 보조 정보를 irt_data에 포함시킨다.
        if irt_data_list["alpha_c_raw"]:
            irt_data["alpha_c_raw"] = np.array(irt_data_list["alpha_c_raw"]).squeeze()
        if irt_data_list["alpha_c_prev"]:
            irt_data["alpha_c_prev"] = np.array(irt_data_list["alpha_c_prev"]).squeeze()
        if irt_data_list["alpha_c_decay_factor"]:
            irt_data["alpha_c_decay_factor"] = np.array(
                irt_data_list["alpha_c_decay_factor"]
            ).squeeze()
        if env_reward_type == "adaptive_risk" and irt_data_list.get(
            "env_crisis_levels"
        ):
            irt_data["env_crisis_levels"] = np.array(
                irt_data_list["env_crisis_levels"]
            ).squeeze()
            irt_data["env_kappa"] = np.array(irt_data_list["env_kappa"]).squeeze()
            irt_data["env_delta_sharpe"] = np.array(
                irt_data_list["env_delta_sharpe"]
            ).squeeze()
        if irt_data_list["crisis_levels_pre_guard"]:
            irt_data["crisis_levels_pre_guard"] = np.array(
                irt_data_list["crisis_levels_pre_guard"]
            ).squeeze()
        if irt_data_list["crisis_raw"]:
            irt_data["crisis_raw"] = np.array(irt_data_list["crisis_raw"]).squeeze()
        if irt_data_list["crisis_bias"]:
            irt_data["crisis_bias"] = np.array(irt_data_list["crisis_bias"]).squeeze()
        if irt_data_list["crisis_temperature"]:
            irt_data["crisis_temperature"] = np.array(
                irt_data_list["crisis_temperature"]
            ).squeeze()
        if irt_data_list["crisis_guard_rate"]:
            irt_data["crisis_guard_rate"] = np.array(
                irt_data_list["crisis_guard_rate"]
            ).squeeze()

        env_info = {
            "reward_type": env_reward_type,
            "reward_scaling": getattr(base_env, "reward_scaling", None),
            "use_weighted_action": getattr(base_env, "use_weighted_action", None),
            "weight_slippage": getattr(base_env, "weight_slippage", None),
            "weight_transaction_cost": getattr(
                base_env, "weight_transaction_cost", None
            ),
            "adaptive_lambda_sharpe": getattr(base_env, "adaptive_lambda_sharpe", None),
            "adaptive_lambda_cvar": getattr(base_env, "adaptive_lambda_cvar", None),
            "adaptive_lambda_turnover": getattr(
                base_env, "adaptive_lambda_turnover", None
            ),
            "adaptive_crisis_gain_sharpe": getattr(
                base_env, "adaptive_crisis_gain_sharpe", None
            ),
            "adaptive_crisis_gain_cvar": getattr(
                base_env, "adaptive_crisis_gain_cvar", None
            ),
            "adaptive_dsr_beta": getattr(base_env, "adaptive_dsr_beta", None),
            "adaptive_cvar_window": getattr(base_env, "adaptive_cvar_window", None),
        }
        irt_data["env_info"] = env_info
        if env_meta:
            irt_data["env_meta"] = env_meta

    # 성능 지표 계산
    weights_history = irt_data["weights"] if irt_data else None
    executed_weights_history = irt_data["actual_weights"] if irt_data else None
    returns_for_metrics = (
        execution_returns_array if cap_metrics else execution_returns_raw
    )
    metrics = calculate_metrics(
        portfolio_values,
        initial_amount,
        weights_history,
        returns=returns_for_metrics,
        executed_weights_history=executed_weights_history,
        turnover_target_series=turnover_target_array,
    )

    avg_turnover_exec = (
        float(np.mean(turnover_executed_array)) if turnover_executed_array.size else 0.0
    )
    avg_turnover_target = (
        float(np.mean(turnover_target_array)) if turnover_target_array.size else 0.0
    )

    metrics["avg_turnover"] = avg_turnover_exec
    metrics["avg_turnover_executed"] = avg_turnover_exec
    metrics["avg_turnover_target_env"] = avg_turnover_target
    metrics["avg_turnover_target"] = avg_turnover_target
    metrics["turnover_executed_std"] = (
        float(np.std(turnover_executed_array)) if turnover_executed_array.size else 0.0
    )
    metrics["turnover_target_std"] = (
        float(np.std(turnover_target_array)) if turnover_target_array.size else 0.0
    )
    metrics["turnover_transfer_ratio"] = (
        float(avg_turnover_exec / (avg_turnover_target + 1e-8))
        if avg_turnover_exec or avg_turnover_target
        else 0.0
    )

    metrics["returns_capped_for_metrics"] = bool(cap_metrics)
    if execution_returns_array.size and execution_returns_raw.size:
        diff = execution_returns_array - execution_returns_raw
        diff_abs = np.abs(diff)
        metrics["sanitize_gap_mean"] = float(np.mean(diff_abs))
        metrics["sanitize_gap_max"] = float(np.max(diff_abs))
    else:
        metrics["sanitize_gap_mean"] = 0.0
        metrics["sanitize_gap_max"] = 0.0

    if irt_data is not None:

        def _safe_array(name):
            value = irt_data.get(name)
            if value is None:
                return None
            arr = np.asarray(value)
            return arr if arr.size else None

        alpha_vals = _safe_array("alpha_c")
        if alpha_vals is not None:
            metrics["alpha_c_mean_eval"] = float(np.mean(alpha_vals))
            metrics["alpha_c_std_eval"] = float(np.std(alpha_vals))
            metrics["alpha_c_min_eval"] = float(np.min(alpha_vals))
            metrics["alpha_c_max_eval"] = float(np.max(alpha_vals))

        # α 원본 값과 이전 값에 대한 통계를 추가 계산한다.
        alpha_raw_vals = _safe_array("alpha_c_raw")
        alpha_prev_vals = _safe_array("alpha_c_prev")
        if alpha_raw_vals is not None:
            metrics["alpha_c_raw_mean_eval"] = float(np.mean(alpha_raw_vals))
            metrics["alpha_c_raw_std_eval"] = float(np.std(alpha_raw_vals))
        if alpha_prev_vals is not None:
            metrics["alpha_c_prev_mean_eval"] = float(np.mean(alpha_prev_vals))
        if (
            alpha_vals is not None
            and alpha_prev_vals is not None
            and len(alpha_vals) == len(alpha_prev_vals)
        ):
            delta_alpha = alpha_vals - alpha_prev_vals
            metrics["delta_alpha_c_mean"] = float(np.mean(delta_alpha))
            metrics["delta_alpha_c_std"] = float(np.std(delta_alpha))
            metrics["delta_alpha_c_abs_mean"] = float(np.mean(np.abs(delta_alpha)))

        crisis_vals = _safe_array("crisis_levels")
        if crisis_vals is not None:
            metrics["crisis_level_max_eval"] = float(np.max(crisis_vals))
            metrics["crisis_level_p90_eval"] = float(np.quantile(crisis_vals, 0.9))
            metrics["crisis_level_p99_eval"] = float(np.quantile(crisis_vals, 0.99))
            metrics["crisis_level_median_eval"] = float(np.median(crisis_vals))
            metrics["crisis_activation_rate"] = float(np.mean(crisis_vals >= 0.55))

        # 레짐 분류 결과를 활용해 위기·평시 구간별 통계를 산출한다.
        crisis_regime_vals = _safe_array("crisis_regime")
        if crisis_regime_vals is not None and len(crisis_regime_vals) > 0:
            crisis_mask = crisis_regime_vals == 1
            normal_mask = crisis_regime_vals == 0
            crisis_steps = int(np.sum(crisis_mask))
            normal_steps = int(np.sum(normal_mask))
            total_steps = len(crisis_regime_vals)

            metrics["crisis_regime_pct"] = float(crisis_steps / max(total_steps, 1))
            metrics["crisis_steps"] = crisis_steps
            metrics["normal_steps"] = normal_steps

            # 레짐별 수익률/샤프 분리 (execution_returns_array 사용)
            if len(execution_returns_array) == len(crisis_regime_vals):
                if crisis_steps > 0:
                    crisis_returns = execution_returns_array[crisis_mask]
                    metrics["crisis_mean_return"] = float(np.mean(crisis_returns))
                    metrics["crisis_volatility"] = float(np.std(crisis_returns))
                    if metrics["crisis_volatility"] > 1e-8:
                        metrics["crisis_sharpe"] = float(
                            metrics["crisis_mean_return"]
                            / metrics["crisis_volatility"]
                            * np.sqrt(252)
                        )
                if normal_steps > 0:
                    normal_returns = execution_returns_array[normal_mask]
                    metrics["normal_mean_return"] = float(np.mean(normal_returns))
                    metrics["normal_volatility"] = float(np.std(normal_returns))
                    if metrics["normal_volatility"] > 1e-8:
                        metrics["normal_sharpe"] = float(
                            metrics["normal_mean_return"]
                            / metrics["normal_volatility"]
                            * np.sqrt(252)
                        )

        reward_components_dict = irt_data.get("reward_components")
        if reward_components_dict:
            component_items = [
                (key, np.asarray(values, dtype=np.float64))
                for key, values in sorted(reward_components_dict.items())
                if len(values) > 0
            ]
            if component_items:
                component_matrix = np.vstack([values for _, values in component_items])
                abs_sum = np.sum(np.abs(component_matrix), axis=0)
                abs_sum = np.where(abs_sum <= 1e-12, 1e-12, abs_sum)
                for idx, (key, values) in enumerate(component_items):
                    metrics[f"reward_component_{key}_mean"] = float(np.mean(values))
                    metrics[f"reward_component_{key}_std"] = float(np.std(values))
                    shares = np.abs(component_matrix[idx]) / abs_sum
                    metrics[f"reward_component_{key}_abs_share"] = float(
                        np.mean(shares)
                    )

        hyst_up_vals = _safe_array("hysteresis_up")
        hyst_down_vals = _safe_array("hysteresis_down")
        if (
            hyst_up_vals is not None
            and hyst_down_vals is not None
            and len(hyst_up_vals) > 0
            and len(hyst_down_vals) > 0
        ):
            metrics["hysteresis_up_mean"] = float(np.mean(hyst_up_vals))
            metrics["hysteresis_down_mean"] = float(np.mean(hyst_down_vals))
            # 히스테리시스 상·하한 및 폭을 함께 기록한다.
            metrics["hysteresis_up_min"] = float(np.min(hyst_up_vals))
            metrics["hysteresis_up_max"] = float(np.max(hyst_up_vals))
            metrics["hysteresis_down_min"] = float(np.min(hyst_down_vals))
            metrics["hysteresis_down_max"] = float(np.max(hyst_down_vals))
            metrics["hysteresis_width_mean"] = float(
                np.mean(hyst_up_vals - hyst_down_vals)
            )

        proto_weights = _safe_array("prototype_weights")
        if proto_weights is not None:
            weights_clipped = np.clip(proto_weights, 1e-12, 1.0)
            proto_entropy = -np.sum(weights_clipped * np.log(weights_clipped), axis=1)
            proto_max = np.max(weights_clipped, axis=1)
            proto_var = np.var(weights_clipped, axis=1)
            metrics["prototype_entropy_mean_eval"] = float(np.mean(proto_entropy))
            metrics["prototype_entropy_std_eval"] = float(np.std(proto_entropy))
            metrics["prototype_entropy_min_eval"] = float(np.min(proto_entropy))
            metrics["prototype_entropy_max_eval"] = float(np.max(proto_entropy))
            metrics["prototype_max_weight_mean_eval"] = float(np.mean(proto_max))
            metrics["prototype_max_weight_max_eval"] = float(np.max(proto_max))
            metrics["prototype_var_mean_eval"] = float(np.mean(proto_var))

    xai_dir_path: Optional[Path] = None
    if xai_enabled:
        if xai_output_dir:
            xai_dir_path = Path(xai_output_dir).expanduser()
        elif "model_path_obj" in locals() and model_path_obj is not None:
            xai_dir_path = model_path_obj.parent / "xai"
        else:
            xai_dir_path = Path(model_path).expanduser().resolve().parent / "xai"
        xai_dir_path.mkdir(parents=True, exist_ok=True)

        if xai_proto_records:
            base_keys = [
                "step",
                "proto_entropy",
                "alpha_c_mean",
                "alpha_c_std",
                "crisis_level",
                "action_temp",
                "cash_weight",
                "sum_weights",
                "kappa_sharpe",
                "kappa_cvar",
                "turnover_executed",
            ]
            proto_rows: List[Dict[str, Any]] = []
            for rec in xai_proto_records:
                row: Dict[str, Any] = {}
                for key in base_keys:
                    if key == "step":
                        row[key] = int(rec.get("step", 0))
                    else:
                        val = rec.get(key)
                        row[key] = float(val) if val is not None else np.nan
                top_indices = rec.get("top_indices", []) or []
                top_weights = rec.get("top_weights", []) or []
                for idx, proto_idx in enumerate(top_indices, start=1):
                    row[f"topk_{idx}_id"] = int(proto_idx)
                    weight_val = (
                        float(top_weights[idx - 1])
                        if idx - 1 < len(top_weights)
                        else 0.0
                    )
                    row[f"topk_{idx}_weight"] = weight_val
                if top_weights:
                    other_weight = max(0.0, 1.0 - float(np.sum(top_weights)))
                    row["topk_other_weight"] = float(other_weight)
                proto_rows.append(row)
            proto_df = pd.DataFrame(proto_rows)
            proto_df.to_csv(xai_dir_path / "xai_prototypes_timeseries.csv", index=False)

    if xai_dir_path is not None and xai_collect_features and xai_feature_samples:
        feature_names_list: Optional[List[str]] = None
        if feature_columns and xai_feature_samples:
            obs_dim = len(xai_feature_samples[0]["obs"])
            if len(feature_columns) == obs_dim:
                feature_names_list = list(feature_columns)
        if feature_names_list is None and xai_feature_samples:
            obs_dim = len(xai_feature_samples[0]["obs"])
            feature_names_list = [f"feature_{i}" for i in range(obs_dim)]

        if hasattr(model.policy, "actor"):
            actor_prev_state = model.policy.actor.training
            model.policy.actor.eval()
        else:
            actor_prev_state = None
        if hasattr(model, "critic"):
            critic_prev_state = model.critic.training
            model.critic.eval()
        else:
            critic_prev_state = None

        def _forward_value(
            target_name: str,
            obs_tensor: torch.Tensor,
            action_tensor: torch.Tensor,
        ) -> torch.Tensor:
            if target_name == "critic_q" and hasattr(model, "critic"):
                q1, q2 = model.critic(obs_tensor, action_tensor)
                return torch.min(q1, q2)
            # log_prob: 정책 확률 관점 (Dirichlet 기반); actor.action_log_prob는
            # 정책이 생성한 확률 질량의 로그를 반환한다.
            fitness = model.policy.actor._compute_fitness(
                obs_tensor, requires_grad=True
            )
            _, info_grad = model.policy.actor.irt_actor(
                state=obs_tensor,
                fitness=fitness,
                deterministic=True,
                retain_grad=True,
            )
            mixed_conc_grad = info_grad.get("mixed_conc_grad")
            if mixed_conc_grad is None:
                mixed_conc_grad = info_grad.get("mixed_conc_clamped")
            action_temp_val = info_grad.get("action_temp", 1.0)
            logits = mixed_conc_grad / max(float(action_temp_val), 1e-6)
            log_probs = torch.log_softmax(logits, dim=-1)
            action_epsilon = 1e-8
            action_clipped = torch.clamp(action_tensor, min=action_epsilon)
            action_norm = action_clipped / action_clipped.sum(dim=-1, keepdim=True)
            return torch.sum(action_norm * log_probs, dim=-1)

        for sample in xai_feature_samples:
            obs_np = np.asarray(sample["obs"], dtype=np.float32)
            action_np = np.asarray(sample["action"], dtype=np.float32)
            for target_name in xai_targets_to_use:
                obs_tensor = torch.from_numpy(obs_np).to(model.device).unsqueeze(0)
                action_tensor = (
                    torch.from_numpy(action_np).to(model.device).unsqueeze(0)
                )

                if xai_ig_steps <= 1:
                    obs_tensor = obs_tensor.clone().detach().requires_grad_(True)
                    target_val = _forward_value(target_name, obs_tensor, action_tensor)
                    grad = torch.autograd.grad(target_val, obs_tensor)[0]
                    attribution = (grad * obs_tensor).squeeze(0).detach().cpu().numpy()
                else:
                    baseline = torch.zeros_like(obs_tensor)
                    obs_diff = obs_tensor - baseline
                    total_attr = torch.zeros_like(obs_tensor)
                    for alpha in np.linspace(0.0, 1.0, xai_ig_steps):
                        blended = (
                            (baseline + alpha * obs_diff)
                            .clone()
                            .detach()
                            .requires_grad_(True)
                        )
                        target_val = _forward_value(target_name, blended, action_tensor)
                        grad = torch.autograd.grad(target_val, blended)[0]
                        total_attr += grad * obs_diff
                    attribution = (
                        (total_attr / float(xai_ig_steps))
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                abs_attr = np.abs(attribution)
                top_count = min(abs_attr.size, 10)
                top_indices = np.argsort(abs_attr)[::-1][:top_count]
                regime_label = "crisis" if sample["crisis_level"] >= 0.55 else "normal"
                for rank, idx in enumerate(top_indices, start=1):
                    feature_name = (
                        feature_names_list[idx]
                        if feature_names_list and idx < len(feature_names_list)
                        else f"feature_{idx}"
                    )
                    att_val = float(attribution[idx])
                    xai_feature_rows.append(
                        {
                            "step": int(sample["step"]),
                            "feature_index": int(idx),
                            "feature_name": feature_name,
                            "attribution": att_val,
                            "abs_attribution": float(abs_attr[idx]),
                            "sign": "positive" if att_val >= 0 else "negative",
                            "crisis_level": float(sample["crisis_level"]),
                            "regime": regime_label,
                            "rank": rank,
                            "target": target_name,
                        }
                    )

        if actor_prev_state is not None and hasattr(model.policy, "actor"):
            model.policy.actor.train(actor_prev_state)
        if critic_prev_state is not None and hasattr(model, "critic"):
            model.critic.train(critic_prev_state)

        feature_df = pd.DataFrame(xai_feature_rows)
        if not feature_df.empty:
            parquet_path = xai_dir_path / "xai_feature_attributions.parquet"
            try:
                feature_df.to_parquet(parquet_path, index=False)
            except (ImportError, ValueError):
                fallback_path = parquet_path.with_suffix(".csv")
                feature_df.to_csv(fallback_path, index=False)

    if xai_dir_path is not None and xai_collect_features:
        crisis_threshold = 0.55
        proto_weights_full = None
        crisis_levels_full = None
        if is_irt and "irt_data" in locals() and isinstance(irt_data, dict):
            proto_weights_full = irt_data.get("prototype_weights")
            crisis_levels_full = irt_data.get("crisis_levels")

        def _mean_safe(values: np.ndarray) -> Optional[float]:
            if values.size == 0:
                return None
            finite_vals = values[np.isfinite(values)]
            if finite_vals.size == 0:
                return None
            return float(np.mean(finite_vals))

        def _proto_summary(
            weights_array: Optional[np.ndarray], mask: np.ndarray, limit: int = 5
        ) -> List[Dict[str, float]]:
            if (
                weights_array is None
                or weights_array.size == 0
                or mask.size == 0
                or not mask.any()
            ):
                return []
            masked = weights_array[mask]
            if masked.size == 0:
                return []
            avg_weights = masked.mean(axis=0)
            top_idx = np.argsort(avg_weights)[::-1][:limit]
            total = float(np.sum(avg_weights))
            summary_rows: List[Dict[str, float]] = []
            for idx in top_idx:
                mean_weight = float(avg_weights[idx])
                share = float(mean_weight / total) if total > 0 else 0.0
                summary_rows.append(
                    {
                        "index": int(idx),
                        "mean_weight": mean_weight,
                        "weight_share": share,
                    }
                )
            return summary_rows

        def _entropy_mean(
            records: List[Dict[str, Any]], regime: str
        ) -> Optional[float]:
            if not records:
                return None
            values = [
                rec["proto_entropy"]
                for rec in records
                if np.isfinite(rec.get("proto_entropy", np.nan))
                and (
                    (rec["crisis_level"] >= crisis_threshold)
                    if regime == "crisis"
                    else (rec["crisis_level"] < crisis_threshold)
                )
            ]
            return _mean_safe(np.array(values, dtype=np.float64)) if values else None

        def _action_temp_mean(
            records: List[Dict[str, Any]], regime: str
        ) -> Optional[float]:
            if not records:
                return None
            values = [
                rec["action_temp"]
                for rec in records
                if np.isfinite(rec.get("action_temp", np.nan))
                and (
                    (rec["crisis_level"] >= crisis_threshold)
                    if regime == "crisis"
                    else (rec["crisis_level"] < crisis_threshold)
                )
            ]
            return _mean_safe(np.array(values, dtype=np.float64)) if values else None

        def _feature_summary(
            rows: List[Dict[str, Any]], regime: str, limit: int = 10
        ) -> List[Dict[str, Any]]:
            if not rows:
                return []
            agg = defaultdict(float)
            signed = defaultdict(list)
            for row in rows:
                if regime != "overall" and row["regime"] != regime:
                    continue
                key = (row["feature_index"], row["feature_name"])
                agg[key] += float(row["abs_attribution"])
                signed[key].append(float(row["attribution"]))
            if not agg:
                return []
            items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:limit]
            summary_rows: List[Dict[str, Any]] = []
            for (idx, name), total in items:
                feature_name = name if name is not None else f"feature_{idx}"
                contributions = signed[(idx, name)]
                mean_signed = float(np.mean(contributions)) if contributions else 0.0
                summary_rows.append(
                    {
                        "feature_index": int(idx),
                        "feature_name": feature_name,
                        "total_abs_attribution": float(total),
                        "mean_attribution": mean_signed,
                    }
                )
            return summary_rows

        def _cash_mean(series: List[Dict[str, float]], regime: str) -> Optional[float]:
            if not series:
                return None
            values = []
            for entry in series:
                value = entry.get("cash_weight")
                if value is None or not np.isfinite(value):
                    continue
                crisis_val = entry.get("crisis_level", 0.0)
                if regime == "normal" and crisis_val >= crisis_threshold:
                    continue
                if regime == "crisis" and crisis_val < crisis_threshold:
                    continue
                values.append(float(value))
            if not values:
                return None
            return float(np.mean(values))

        def _record_count(records: List[Dict[str, Any]], regime: str) -> int:
            if not records:
                return 0
            if regime == "overall":
                return len(records)
            count = 0
            for rec in records:
                crisis_val = rec.get("crisis_level", 0.0)
                if regime == "crisis" and crisis_val >= crisis_threshold:
                    count += 1
                elif regime == "normal" and crisis_val < crisis_threshold:
                    count += 1
            return count

        summary: Dict[str, Any] = {
            "params": {
                "xai_level": xai_level_normalized,
                "xai_target": xai_target_normalized,
                "xai_targets": xai_targets_to_use,
                "xai_log_interval": xai_log_interval,
                "xai_k": xai_k,
                "xai_ig_steps": xai_ig_steps,
                "proto_records": len(xai_proto_records),
                "feature_samples": len(xai_feature_samples),
                "feature_rows": len(xai_feature_rows),
            }
        }

        regimes = {}
        overall_masks = {
            "overall": slice(None),
            "normal": None,
            "crisis": None,
        }
        if (
            proto_weights_full is not None
            and proto_weights_full.size
            and crisis_levels_full is not None
        ):
            crisis_array_full = np.asarray(
                crisis_levels_full, dtype=np.float64
            ).reshape(-1)
            overall_masks["normal"] = crisis_array_full < crisis_threshold
            overall_masks["crisis"] = crisis_array_full >= crisis_threshold

        for regime_name in ("normal", "crisis"):
            avg_cash = _cash_mean(xai_cash_series, regime_name)
            proto_mask = overall_masks.get(regime_name)
            proto_summary = (
                _proto_summary(proto_weights_full, proto_mask)
                if proto_mask is not None
                else []
            )
            regimes[regime_name] = {
                "steps": (
                    int(proto_mask.sum())
                    if isinstance(proto_mask, np.ndarray)
                    else _record_count(xai_proto_records, regime_name)
                ),
                "avg_cash_weight": avg_cash,
                "proto_entropy_mean": _entropy_mean(xai_proto_records, regime_name),
                "action_temp_mean": _action_temp_mean(xai_proto_records, regime_name),
                "top_prototypes": proto_summary,
                "top_features": _feature_summary(xai_feature_rows, regime_name),
            }

        regimes["overall"] = {
            "steps": (
                int(proto_weights_full.shape[0])
                if isinstance(proto_weights_full, np.ndarray)
                else _record_count(xai_proto_records, "overall")
            ),
            "avg_cash_weight": _cash_mean(xai_cash_series, "overall"),
            "proto_entropy_mean": (
                _mean_safe(
                    np.array(
                        [rec["proto_entropy"] for rec in xai_proto_records],
                        dtype=np.float64,
                    )
                )
                if xai_proto_records
                else None
            ),
            "action_temp_mean": (
                _mean_safe(
                    np.array(
                        [rec["action_temp"] for rec in xai_proto_records],
                        dtype=np.float64,
                    )
                )
                if xai_proto_records
                else None
            ),
            "top_prototypes": (
                _proto_summary(
                    proto_weights_full, np.ones(proto_weights_full.shape[0], dtype=bool)
                )
                if isinstance(proto_weights_full, np.ndarray)
                else []
            ),
            "top_features": _feature_summary(xai_feature_rows, "overall"),
        }

        summary["regimes"] = regimes

        feature_attr_summary: Dict[str, Dict[str, Any]] = {}
        for target_name in xai_targets_to_use:
            target_rows = [
                row for row in xai_feature_rows if row.get("target") == target_name
            ]
            target_entry: Dict[str, Any] = {}
            for regime_name in ("normal", "crisis", "overall"):
                target_entry[regime_name] = {
                    "count": _record_count(target_rows, regime_name),
                    "top_features": _feature_summary(target_rows, regime_name),
                }
            feature_attr_summary[target_name] = target_entry
        summary["feature_attributions"] = feature_attr_summary

        summary_path = xai_dir_path / "xai_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )

    artefact_bundle = {
        "per_step_returns": execution_returns_array,
        "per_step_returns_raw": execution_returns_raw,
        "value_returns": value_returns_array,
        "value_returns_raw": value_returns_raw,
        "dates": date_records,
        "holdings_timeseries": holdings_records,
        "trades": trade_records,
        "transaction_costs_series": transaction_costs_array,
        "turnover_executed_series": turnover_executed_array,
        "turnover_target_series": turnover_target_array,
        "cash_ratio_series": np.array(cash_ratio_series, dtype=np.float64),
    }

    return (
        portfolio_values,
        execution_returns_array,
        value_returns_array,
        irt_data,
        metrics,
        artefact_bundle,
    )


def evaluate_direct(args, model_name, model_class):
    """Direct 방식: SB3 모델 직접 사용 (args 객체 wrapper)"""

    return evaluate_model(
        model_path=args.model,
        model_class=model_class,
        test_start=args.test_start,
        test_end=args.test_end,
        stock_tickers=DOW_30_TICKER,
        tech_indicators=INDICATORS,
        initial_amount=1000000,
        verbose=True,
        adaptive_lambda_sharpe=args.adaptive_lambda_sharpe,
        adaptive_lambda_cvar=args.adaptive_lambda_cvar,
        adaptive_lambda_turnover=args.adaptive_lambda_turnover,
        adaptive_crisis_gain_sharpe=args.adaptive_crisis_gain_sharpe,
        adaptive_crisis_gain_cvar=args.adaptive_crisis_gain_cvar,
        adaptive_dsr_beta=args.adaptive_dsr_beta,
        adaptive_cvar_window=args.adaptive_cvar_window,
        cap_metrics=not args.no_cap_metrics,
        xai_level=args.xai_level,
        xai_target=args.xai_target,
        xai_k=args.xai_k,
        xai_ig_steps=args.xai_ig_steps,
        xai_log_interval=args.xai_log_interval,
        xai_output_dir=args.xai_output,
    )


def evaluate_drlagent(args, model_name, model_class):
    """DRLAgent 방식: DRLAgent.DRL_prediction() 사용"""

    print(f"\n[평가 방식: DRLAgent - DRL_prediction()]")

    # 1. 데이터 준비
    print(f"\n[1/4] 평가 데이터 다운로드 중...")
    df = YahooDownloader(
        start_date=args.test_start, end_date=args.test_end, ticker_list=DOW_30_TICKER
    ).fetch_data()
    print(f"  다운로드 완료: {df.shape[0]}행")

    # 2. Feature Engineering
    print(f"\n[2/4] 피처 엔지니어링 수행 중...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        user_defined_feature=False,
    )
    df_processed = fe.preprocess_data(df)
    test_df = data_split(df_processed, args.test_start, args.test_end)
    print(f"  평가 샘플 수: {len(test_df)}")

    # 3. 환경 생성 및 모델 로드
    print(f"\n[3/4] 모델 및 환경을 준비합니다...")
    stock_dim = len(test_df.tic.unique())
    print(f"  실제 사용 종목 수: {stock_dim}")

    # IRT 모델은 dsr_cvar 보상을 사용하므로 자동으로 보상 타입을 전환한다.
    is_irt = "irt" in args.model.lower()
    reward_type = "dsr_cvar" if is_irt else "basic"

    # 상태 공간: 잔고(1) + 가격(N) + 보유량(N) + 기술지표(K*N); dsr_cvar 사용 시 추가 특징이 포함된다.
    state_space = 1 + (len(INDICATORS) + 2) * stock_dim

    env_kwargs = {
        "df": test_df,
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": INDICATORS,
        "print_verbosity": 500,
        # 리스크 민감 보상 설정
        "reward_type": reward_type,
        "lambda_dsr": 0.1,
        "lambda_cvar": 0.05,
    }

    e_test_gym = StockTradingEnv(**env_kwargs)

    model = model_class.load(args.model)
    print(f"  모델 로드에 성공했습니다")
    if is_irt:
        print(f"  reward_type: {reward_type}")

    # 4. DRL_prediction 실행
    print(f"\n[4/4] DRL_prediction 실행 중...")
    account_memory, actions_memory = DRLAgent.DRL_prediction(
        model=model, environment=e_test_gym, deterministic=True
    )

    print(f"  평가 완료: 총 {len(account_memory)} 스텝")

    # account_memory에서 portfolio_values 추출
    portfolio_values = account_memory["account_value"].tolist()

    # DRLAgent 방식은 IRT 데이터 수집 불가
    pv_array = np.asarray(portfolio_values, dtype=np.float64)
    if pv_array.size > 1:
        prev_vals = np.clip(pv_array[:-1], 1e-8, None)
        raw_returns = (pv_array[1:] - pv_array[:-1]) / prev_vals
    else:
        raw_returns = np.array([], dtype=np.float64)
    from finrl.evaluation.visualizer import sanitize_returns

    exec_returns = sanitize_returns(raw_returns)
    value_returns = sanitize_returns(raw_returns)

    cap_metrics = not args.no_cap_metrics
    returns_for_metrics = exec_returns if cap_metrics else raw_returns
    metrics = calculate_metrics(
        portfolio_values,
        returns=returns_for_metrics,
    )
    metrics["returns_capped_for_metrics"] = bool(cap_metrics)
    if exec_returns.size and raw_returns.size:
        diff = exec_returns - raw_returns
        diff_abs = np.abs(diff)
        metrics["sanitize_gap_mean"] = float(np.mean(diff_abs))
        metrics["sanitize_gap_max"] = float(np.max(diff_abs))
    else:
        metrics["sanitize_gap_mean"] = 0.0
        metrics["sanitize_gap_max"] = 0.0

    zero_series = np.zeros_like(exec_returns)
    artefact_bundle = {
        "per_step_returns": exec_returns,
        "per_step_returns_raw": raw_returns,
        "value_returns": value_returns,
        "value_returns_raw": raw_returns,
        "dates": [],
        "holdings_timeseries": [],
        "trades": [],
        "transaction_costs_series": zero_series,
        "turnover_executed_series": zero_series,
        "turnover_target_series": zero_series,
        "cash_ratio_series": zero_series,
    }

    return portfolio_values, exec_returns, value_returns, None, metrics, artefact_bundle


def main(args):
    print("=" * 70)
    print("모델 평가")
    print("=" * 70)

    # 모델 타입 감지
    if args.model_type is None:
        model_name, model_class = detect_model_type(args.model)
        print(f"\n  자동 감지된 모델 유형: {model_name.upper()}")
    else:
        model_name = args.model_type
        model_class = {"sac": SAC, "ppo": PPO, "a2c": A2C, "td3": TD3, "ddpg": DDPG}[
            model_name
        ]

    print(f"\n[평가 설정]")
    print(f"  모델 경로: {args.model}")
    print(f"  모델 유형: {model_name.upper()}")
    print(f"  평가 방식: {args.method}")
    print(f"  평가 구간: {args.test_start} ~ {args.test_end}")

    if args.xai_level != "off" and args.method != "direct":
        print("\n[알림] XAI 출력은 direct 방식 평가에서만 지원되어 비활성화합니다.")
        args.xai_level = "off"
    elif args.xai_level != "off":
        print(
            f"  XAI 설정: level={args.xai_level}, target={args.xai_target}, samples={args.xai_k}, ig_steps={args.xai_ig_steps}"
        )

    # 평가 방식 선택
    if args.method == "direct":
        (
            portfolio_values,
            exec_returns,
            value_returns,
            irt_data,
            metrics,
            artefacts,
        ) = evaluate_direct(args, model_name, model_class)
    elif args.method == "drlagent":
        (
            portfolio_values,
            exec_returns,
            value_returns,
            irt_data,
            metrics,
            artefacts,
        ) = evaluate_drlagent(args, model_name, model_class)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # 6. 결과 출력
    print(f"\n" + "=" * 70)
    print(f"핵심 성능 지표")
    print("=" * 70)
    print(f"\n[평가 기간]")
    print(f"  시작일: {args.test_start}")
    print(f"  종료일: {args.test_end}")
    print(f"  스텝 수: {metrics['n_steps']}")

    print(f"\n[수익 지표]")
    print(f"  총 수익률: {metrics['total_return']*100:.2f}%")
    print(f"  연환산 수익률: {metrics['annualized_return']*100:.2f}%")

    print(f"\n[위험 지표]")
    print(f"  연환산 변동성: {metrics['volatility']*100:.2f}%")
    print(f"  최대 손실폭: {metrics['max_drawdown']*100:.2f}%")

    print(f"\n[위험조정 수익]")
    print(f"  샤프 지수: {metrics['sharpe_ratio']:.3f}")
    print(f"  소르티노 지수: {metrics['sortino_ratio']:.3f}")
    print(f"  칼마 지수: {metrics['calmar_ratio']:.3f}")

    print(f"\n[포트폴리오 가치]")
    print(f"  초기 자산: ${1000000:,.2f}")
    print(f"  최종 자산: ${metrics['final_value']:,.2f}")
    print(f"  손익: ${metrics['final_value'] - 1000000:,.2f}")

    print(f"\n" + "=" * 70)

    # 7. 시각화 (기본 활성화)
    if not args.no_plot:
        output_dir = args.output or os.path.join(
            os.path.dirname(args.model), "evaluation_plots"
        )
        plot_results(
            portfolio_values,
            exec_returns,
            output_dir,
            model_name=model_name.upper(),
            model_path=args.model,
            irt_data=irt_data,
        )

    env_info = {}
    env_meta = None
    if isinstance(irt_data, dict):
        env_info = irt_data.pop("env_info", {}) or {}
        env_meta = irt_data.pop("env_meta", None)

    env_reward_type = env_info.get("reward_type")
    reward_scaling = env_info.get("reward_scaling")
    use_weighted_action = env_info.get("use_weighted_action")
    weight_slippage = env_info.get("weight_slippage")
    weight_transaction_cost = env_info.get("weight_transaction_cost")
    adaptive_lambda_sharpe = env_info.get("adaptive_lambda_sharpe")
    adaptive_lambda_cvar = env_info.get("adaptive_lambda_cvar")
    adaptive_lambda_turnover = env_info.get("adaptive_lambda_turnover")
    adaptive_crisis_gain_sharpe = env_info.get("adaptive_crisis_gain_sharpe")
    adaptive_crisis_gain_cvar = env_info.get("adaptive_crisis_gain_cvar")
    adaptive_dsr_beta = env_info.get("adaptive_dsr_beta")
    adaptive_cvar_window = env_info.get("adaptive_cvar_window")

    if adaptive_lambda_sharpe is None:
        adaptive_lambda_sharpe = args.adaptive_lambda_sharpe
    if adaptive_lambda_cvar is None:
        adaptive_lambda_cvar = args.adaptive_lambda_cvar
    if adaptive_lambda_turnover is None:
        adaptive_lambda_turnover = args.adaptive_lambda_turnover
    if adaptive_crisis_gain_sharpe is None:
        adaptive_crisis_gain_sharpe = args.adaptive_crisis_gain_sharpe
    if adaptive_crisis_gain_cvar is None:
        adaptive_crisis_gain_cvar = args.adaptive_crisis_gain_cvar
    if adaptive_dsr_beta is None:
        adaptive_dsr_beta = args.adaptive_dsr_beta
    if adaptive_cvar_window is None:
        adaptive_cvar_window = args.adaptive_cvar_window

    artefact_dir: Optional[Path] = None
    if not args.no_json:
        from finrl.evaluation.visualizer import save_evaluation_results

        artefact_dir = Path(
            args.output_json or os.path.dirname(args.model) or "."
        ).expanduser()

        series_payload = {
            "portfolio_values": np.asarray(portfolio_values, dtype=np.float64),
            "per_step_returns": artefacts.get("per_step_returns"),
            "per_step_returns_raw": artefacts.get("per_step_returns_raw"),
            "value_returns": artefacts.get("value_returns"),
            "value_returns_raw": artefacts.get("value_returns_raw"),
            "transaction_costs": artefacts.get("transaction_costs_series"),
            "turnover_executed": artefacts.get("turnover_executed_series"),
            "turnover_target": artefacts.get("turnover_target_series"),
            "cash_ratio": artefacts.get("cash_ratio_series"),
            "dates": artefacts.get("dates"),
        }

        tables_payload = {
            "holdings_timeseries": artefacts.get("holdings_timeseries"),
            "trades": artefacts.get("trades"),
        }

        base_results: Dict[str, Any] = {
            "model": {
                "path": args.model,
                "type": model_name,
                "evaluation_method": args.method,
            },
            "test_period": {
                "start": args.test_start,
                "end": args.test_end,
                "steps": metrics["n_steps"],
            },
            "metrics": metrics,
            "series": series_payload,
            "tables": tables_payload,
        }

        if irt_data is not None:
            base_results["irt"] = irt_data

        config = None
        if irt_data is not None:
            config = {
                "env": {
                    "reward_type": env_reward_type,
                    "reward_scaling": reward_scaling,
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
                },
                "meta": env_meta,
            }

        save_evaluation_results(base_results, artefact_dir, config)

    if args.viz != "off":
        if artefact_dir is None:
            print("[viz] 시각화를 건너뜁니다: JSON 결과가 생성되지 않았습니다.")
        else:
            run_dir = artefact_dir
            out_dir = (
                Path(args.viz_outdir).expanduser()
                if args.viz_outdir
                else run_dir / "viz"
            )
            include_core = args.viz in ("core", "all")
            include_xai = args.viz in ("xai", "all")
            cmd = [
                sys.executable,
                "scripts/visualize_from_json.py",
                "--input-dir",
                str(run_dir),
                "--out-dir",
                str(out_dir),
                "--format",
                args.viz_format,
                "--dpi",
                str(args.viz_dpi),
                "--crisis-threshold",
                str(args.viz_crisis_threshold),
            ]
            cmd.append("--include-core" if include_core else "--no-core")
            cmd.append("--include-xai" if include_xai else "--no-xai")
            try:
                subprocess.run(cmd, check=True)
                print(f"[viz] 시각화 결과가 {out_dir} 위치에 저장되었습니다.")
            except Exception as exc:
                print(f"[viz] 시각화를 건너뜁니다 (오류: {exc})")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 상세 평가")

    parser.add_argument(
        "--model", type=str, required=True, help="모델 파일 경로 (.zip)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["sac", "ppo", "a2c", "td3", "ddpg"],
        help="모델 타입 (자동 감지 실패 시 명시)",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="direct",
        choices=["direct", "drlagent"],
        help="평가 방식 (기본: direct)",
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
    parser.add_argument(
        "--adaptive-lambda-sharpe",
        type=float,
        default=0.20,
        help="적응형 위험 보상 Sharpe 가중치 λ_S (기본: 0.20)",
    )
    parser.add_argument(
        "--adaptive-lambda-cvar",
        type=float,
        default=0.40,
        help="적응형 위험 보상 CVaR 가중치 β (기본: 0.40)",
    )
    parser.add_argument(
        "--adaptive-lambda-turnover",
        type=float,
        default=0.0,
        help="적응형 위험 보상의 회전율 패널티 μ (기본: 0.0; NAV에 거래비용이 미포함인 경우에만 양수로 설정)",
    )
    parser.add_argument(
        "--adaptive-crisis-gain",
        dest="adaptive_crisis_gain_sharpe",
        type=float,
        default=-0.15,
        help="적응형 위험 보상 Sharpe 위기 보정 g_S (기본: -0.15, 음수 유지)",
    )
    parser.add_argument(
        "--adaptive-crisis-gain-cvar",
        type=float,
        default=0.25,
        help="적응형 위험 보상 CVaR 위기 보정 g_C (기본: 0.25)",
    )
    parser.add_argument(
        "--adaptive-dsr-beta",
        type=float,
        default=0.92,
        help="적응형 위험 보상 DSR EMA β (기본: 0.92)",
    )
    parser.add_argument(
        "--adaptive-cvar-window",
        type=int,
        default=40,
        help="적응형 위험 보상 CVaR 윈도우 길이 (기본: 40)",
    )

    parser.set_defaults(no_cap_metrics=True)
    parser.add_argument(
        "--no-cap-metrics",
        dest="no_cap_metrics",
        action="store_true",
        help="메트릭 계산 시 수익률 클리핑 비활성화 (기본값)",
    )
    parser.add_argument(
        "--cap-metrics",
        dest="no_cap_metrics",
        action="store_false",
        help="메트릭 계산 전에 수익률을 클리핑합니다 (추천하지 않음).",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="시각화 결과 저장 비활성화 (기본: 활성화)",
    )
    parser.add_argument(
        "--no-json", action="store_true", help="JSON 결과 저장 비활성화 (기본: 활성화)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="플롯 출력 디렉터리 (기본: 모델 디렉토리/evaluation_plots)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="JSON 출력 파일 (기본: 모델 디렉토리/evaluation_results.json)",
    )
    parser.add_argument(
        "--xai-level",
        type=str,
        default="full",
        choices=["off", "light", "full"],
        help="XAI 산출 수준 (off=생성 안 함, light=프로토타입 시계열, full=전체 번들)",
    )
    parser.add_argument(
        "--xai-target",
        type=str,
        default="both",
        choices=["critic_q", "log_prob", "both"],
        help="특징 기여도 계산 대상 (critic_q: Q(s,a), log_prob: log π(a|s), both: 두 타깃 모두)",
    )
    parser.add_argument(
        "--xai-k",
        type=int,
        default=20,
        help="특징 기여도 샘플링 최대 스텝 수 (기본: 20)",
    )
    parser.add_argument(
        "--xai-ig-steps",
        type=int,
        default=20,
        help="Integrated Gradients 분할 스텝 수 (기본: 20; 1이면 Grad×Input)",
    )
    parser.add_argument(
        "--xai-log-interval",
        type=int,
        default=10,
        help="프로토타입 시계열 기록 간격 (스텝 단위, 기본: 10)",
    )
    parser.add_argument(
        "--xai-output",
        type=str,
        default=None,
        help="XAI 산출물 출력 디렉토리 (기본: 모델 폴더/xai)",
    )
    parser.add_argument(
        "--viz",
        choices=["off", "core", "xai", "all"],
        default="all",
        help="평가 종료 후 자동 시각화 수준",
    )
    parser.add_argument(
        "--viz-outdir",
        type=str,
        default=None,
        help="자동 시각화 출력 디렉토리 (기본: JSON 폴더/viz)",
    )
    parser.add_argument(
        "--viz-crisis-threshold",
        type=float,
        default=0.5,
        help="자동 시각화용 위기 임계값",
    )
    parser.add_argument(
        "--viz-format",
        choices=["png", "pdf"],
        default="png",
        help="자동 시각화 이미지 포맷",
    )
    parser.add_argument(
        "--viz-dpi",
        type=int,
        default=160,
        help="자동 시각화 이미지 DPI",
    )

    args = parser.parse_args()

    metrics = main(args)
