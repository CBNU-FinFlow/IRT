# scripts/compare_models.py

"""
두 모델 비교 분석 스크립트

두 모델(SAC vs IRT 등)의 성능과 지표를 비교 분석한다.

Usage:
    # SAC vs IRT 비교
    python scripts/compare_models.py \
      --model1 logs/sac/xxx/sac_final.zip \
      --model2 logs/irt/xxx/irt_final.zip \
      --output comparison_results

    # 출력
    comparison_results/
    ├── comparison_table.txt       # 성능 지표 테이블
    ├── comparison_summary.json    # JSON 결과
    └── plots/
        ├── portfolio_value_comparison.png
        ├── returns_distribution.png
        ├── drawdown_comparison.png
        ├── performance_metrics.png
        ├── risk_metrics.png
        ├── cumulative_returns.png
        ├── rolling_sharpe.png
        └── crisis_response.png (IRT only)
"""

import argparse
import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List

from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS, TEST_START_DATE, TEST_END_DATE
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import SAC, PPO, A2C, TD3, DDPG


# 타임스탬프 패턴 (YYYYMMDD_HHMMSS)
TIMESTAMP_PATTERN = re.compile(r'\d{8}_\d{6}')


def find_latest_timestamp(base_dir: Path) -> Optional[Path]:
    """
    베이스 디렉토리에서 가장 최근 타임스탬프 디렉토리 찾기

    Args:
        base_dir: 검색할 베이스 디렉토리 (예: logs/sac)

    Returns:
        가장 최근 타임스탬프 디렉토리 Path, 없으면 None
    """
    timestamp_dirs = []
    for d in base_dir.iterdir():
        if d.is_dir() and TIMESTAMP_PATTERN.match(d.name):
            timestamp_dirs.append(d)

    if not timestamp_dirs:
        return None

    # 타임스탬프 정렬 (문자열 정렬로 충분, 최신이 마지막)
    return sorted(timestamp_dirs, key=lambda p: p.name, reverse=True)[0]


def _resolve_in_timestamp_dir(timestamp_dir: Path, use_best: bool) -> str:
    """
    타임스탬프 디렉토리 내에서 모델 파일 찾기

    Args:
        timestamp_dir: 타임스탬프 디렉토리 (예: logs/sac/20251005_123456)
        use_best: True면 best_model, False면 final 모델

    Returns:
        모델 파일 경로

    Raises:
        FileNotFoundError: 모델을 찾을 수 없을 때
    """
    if use_best:
        best_path = timestamp_dir / "best_model" / "best_model.zip"
        if best_path.exists():
            return str(best_path)
        raise FileNotFoundError(f"Best model not found: {best_path}")
    else:
        # *_final.zip 또는 irt_final.zip 찾기
        final_models = list(timestamp_dir.glob("*_final.zip"))
        if not final_models:
            raise FileNotFoundError(f"Final model not found in {timestamp_dir}")
        return str(final_models[0])


def resolve_model_path(path: str, use_best: bool = False) -> str:
    """
    모델 경로를 자동으로 해석

    Args:
        path: 모델 경로 (디렉토리 또는 파일)
        use_best: True면 best_model 사용, False면 final 사용

    Returns:
        실제 모델 파일 경로

    Examples:
        logs/sac → logs/sac/20251005_123456/sac_final.zip (최신)
        logs/sac --use-best → logs/sac/20251005_123456/best_model/best_model.zip
        logs/sac/20251005_123456 → logs/sac/20251005_123456/sac_final.zip
        logs/sac/20251005_123456/sac_final.zip → logs/sac/20251005_123456/sac_final.zip (그대로)

    Raises:
        FileNotFoundError: 경로를 찾을 수 없을 때
        ValueError: 타임스탬프 디렉토리를 찾을 수 없을 때
    """
    p = Path(path)

    # 1. 파일인 경우 - 그대로 반환
    if p.is_file():
        return str(p)

    # 2. 디렉토리인 경우
    if p.is_dir():
        # 타임스탬프 패턴 확인
        if TIMESTAMP_PATTERN.match(p.name):
            # 특정 타임스탬프 디렉토리
            return _resolve_in_timestamp_dir(p, use_best)
        else:
            # 베이스 디렉토리 (logs/sac 등)
            latest = find_latest_timestamp(p)
            if latest is None:
                raise ValueError(f"No timestamp directories found in {p}")
            print(f"  Found latest: {latest.name}")
            return _resolve_in_timestamp_dir(latest, use_best)

    raise FileNotFoundError(f"Path not found: {path}")


def detect_model_type(model_path: str) -> Tuple[str, any]:
    """
    모델 파일명에서 모델 타입 자동 감지

    Returns:
        (model_name, model_class): 예) ('sac', SAC) 또는 ('irt', SAC)
    """
    filename = os.path.basename(model_path).lower()
    path_lower = model_path.lower()

    # IRT 모델 우선 확인 (SAC 기반)
    if 'irt' in filename or 'irt' in path_lower:
        return 'irt', SAC

    if 'sac' in filename or 'sac' in path_lower:
        return 'sac', SAC
    elif 'ppo' in filename or 'ppo' in path_lower:
        return 'ppo', PPO
    elif 'a2c' in filename or 'a2c' in path_lower:
        return 'a2c', A2C
    elif 'td3' in filename or 'td3' in path_lower:
        return 'td3', TD3
    elif 'ddpg' in filename or 'ddpg' in path_lower:
        return 'ddpg', DDPG
    else:
        raise ValueError(
            f"모델 타입을 자동 감지할 수 없습니다: {model_path}\n"
            f"파일명 또는 경로에 모델명(sac/ppo/a2c/td3/ddpg/irt)을 포함하세요."
        )


def create_env(df: pd.DataFrame, stock_dim: int, tech_indicators: List[str]):
    """환경 생성"""
    state_space = 1 + (len(tech_indicators) + 2) * stock_dim

    env_kwargs = {
        "df": df,
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": tech_indicators,
        "print_verbosity": 500
    }

    return StockTradingEnv(**env_kwargs)


def calculate_metrics(portfolio_values: np.ndarray, initial_amount: float = 1000000) -> Dict:
    """
    성능 지표 계산

    Args:
        portfolio_values: 포트폴리오 가치 배열
        initial_amount: 초기 자본

    Returns:
        dict: 성능 지표 딕셔너리
    """
    from finrl.evaluation.metrics import (
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
        calculate_calmar_ratio,
        calculate_max_drawdown,
        calculate_var,
        calculate_cvar
    )

    pv = np.array(portfolio_values)

    # Daily returns
    returns = (pv[1:] - pv[:-1]) / pv[:-1]

    # Total return
    total_return = (pv[-1] - initial_amount) / initial_amount

    # Annualized return
    n_days = len(pv) - 1
    annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

    # Volatility (annualized)
    volatility = np.std(returns) * np.sqrt(252)

    # Sharpe Ratio
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)

    # Maximum Drawdown
    max_drawdown = calculate_max_drawdown(pv)

    # Calmar Ratio
    calmar_ratio = calculate_calmar_ratio(total_return, max_drawdown, periods_per_year=252, n_years=n_days/252)

    # Sortino Ratio
    sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252)

    # VaR and CVaR
    var_95 = calculate_var(returns, alpha=0.05)
    cvar_95 = calculate_cvar(returns, alpha=0.05)

    return {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'calmar_ratio': float(calmar_ratio),
        'max_drawdown': float(max_drawdown),
        'volatility': float(volatility),
        'var_95': float(var_95),
        'cvar_95': float(cvar_95),
        'final_value': float(pv[-1]),
        'initial_value': float(initial_amount)
    }


def evaluate_model(model_path: str, test_start: str, test_end: str) -> Dict:
    """
    모델 평가 실행

    Args:
        model_path: 모델 파일 경로
        test_start: 테스트 시작 날짜
        test_end: 테스트 종료 날짜

    Returns:
        dict: 평가 결과 (metrics, portfolio_values, irt_data 등)
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {os.path.basename(model_path)}")
    print(f"{'='*70}")

    # 1. 모델 타입 감지
    model_name, model_class = detect_model_type(model_path)
    print(f"Model type: {model_name.upper()}")

    # 2. 데이터 준비
    print(f"\n[1/4] Downloading test data...")
    df = YahooDownloader(
        start_date=test_start,
        end_date=test_end,
        ticker_list=DOW_30_TICKER
    ).fetch_data()
    print(f"  Downloaded: {len(df)} rows")

    # 3. Feature Engineering
    print(f"\n[2/4] Feature Engineering...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False
    )
    df = fe.preprocess_data(df)
    print(f"  Test rows: {len(df)}")

    # 4. 환경 생성
    print(f"\n[3/4] Creating environment...")
    stock_dim = len(df.tic.unique())
    test_env = create_env(df, stock_dim, INDICATORS)
    print(f"  Stock dimension: {stock_dim}")

    # 5. 모델 로드
    print(f"\n[4/4] Loading model...")
    model = model_class.load(model_path)
    print(f"  Model loaded successfully")

    # 6. 평가 실행
    print(f"\nRunning evaluation...")
    obs, _ = test_env.reset()
    done = False
    portfolio_values = [1000000]
    total_reward = 0

    # IRT 데이터 수집 (IRT 모델인 경우)
    irt_data_list = {
        'w': [],
        'w_rep': [],
        'w_ot': [],
        'crisis_levels': [],
        'crisis_types': [],
        'cost_matrices': [],
        'weights': []
    }
    is_irt = (model_name == 'irt')

    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)

        # IRT info 수집
        if is_irt and hasattr(model.policy, 'get_irt_info'):
            info_dict = model.policy.get_irt_info()
            if info_dict is not None:
                irt_data_list['w'].append(info_dict['w'][0].cpu().numpy())
                irt_data_list['w_rep'].append(info_dict['w_rep'][0].cpu().numpy())
                irt_data_list['w_ot'].append(info_dict['w_ot'][0].cpu().numpy())
                irt_data_list['crisis_levels'].append(info_dict['crisis_level'][0].cpu().numpy())
                irt_data_list['crisis_types'].append(info_dict['crisis_types'][0].cpu().numpy())
                irt_data_list['cost_matrices'].append(info_dict['cost_matrix'][0].cpu().numpy())

                weights = action / (action.sum() + 1e-8)
                irt_data_list['weights'].append(weights)

        obs, reward, done, truncated, info = test_env.step(action)
        total_reward += reward
        done = done or truncated

        # Portfolio value
        state = np.array(test_env.state)
        cash = state[0]
        prices = state[1:stock_dim+1]
        holdings = state[stock_dim+1:2*stock_dim+1]
        pv = cash + np.sum(prices * holdings)
        portfolio_values.append(pv)

        step += 1

    print(f"  Evaluation completed: {step} steps")

    # 7. IRT 데이터 변환
    irt_data = None
    if is_irt and irt_data_list['w']:
        irt_data = {
            'w_rep': np.array(irt_data_list['w_rep']),
            'w_ot': np.array(irt_data_list['w_ot']),
            'weights': np.array(irt_data_list['weights']),
            'crisis_levels': np.array(irt_data_list['crisis_levels']).squeeze(),
            'crisis_types': np.array(irt_data_list['crisis_types']),
            'prototype_weights': np.array(irt_data_list['w']),
            'cost_matrices': np.array(irt_data_list['cost_matrices']),
        }

    # 8. 메트릭 계산
    metrics = calculate_metrics(portfolio_values)

    print(f"\n[Results]")
    print(f"  Final value: ${metrics['final_value']:,.2f}")
    print(f"  Total return: {metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")

    return {
        'model_name': model_name,
        'model_path': model_path,
        'metrics': metrics,
        'portfolio_values': np.array(portfolio_values),
        'irt_data': irt_data
    }


def print_comparison_table(result1: Dict, result2: Dict):
    """비교 테이블 출력"""
    m1 = result1['metrics']
    m2 = result2['metrics']
    name1 = result1['model_name'].upper()
    name2 = result2['model_name'].upper()

    print(f"\n{'='*70}")
    print(f"{name1} vs {name2} Comparison")
    print(f"{'='*70}\n")

    metrics_to_compare = [
        ('Total Return (%)', 'total_return', 100, '+'),
        ('Annualized Return (%)', 'annualized_return', 100, '+'),
        ('Sharpe Ratio', 'sharpe_ratio', 1, '+'),
        ('Sortino Ratio', 'sortino_ratio', 1, '+'),
        ('Calmar Ratio', 'calmar_ratio', 1, '+'),
        ('Max Drawdown (%)', 'max_drawdown', 100, '-'),
        ('Volatility (%)', 'volatility', 100, '-'),
        ('VaR 95% (%)', 'var_95', 100, '-'),
        ('CVaR 95% (%)', 'cvar_95', 100, '-'),
        ('Final Value ($)', 'final_value', 1, '+')
    ]

    print(f"{'Metric':<25} {name1:>12} {name2:>12} {'Δ%':>10} {'Winner':>8}")
    print(f"{'-'*70}")

    for metric_name, key, scale, direction in metrics_to_compare:
        val1 = m1[key] * scale
        val2 = m2[key] * scale

        if val1 != 0:
            delta_pct = ((val2 - val1) / abs(val1)) * 100
        else:
            delta_pct = 0

        # 승자 결정
        if direction == '+':
            winner = name2 if val2 > val1 else name1 if val1 > val2 else 'Tie'
        else:
            winner = name2 if val2 < val1 else name1 if val1 < val2 else 'Tie'

        winner_mark = '✅' if winner == name2 else ''

        print(f"{metric_name:<25} {val1:>12.4f} {val2:>12.4f} {delta_pct:>9.2f}% {winner_mark:>8}")

    print(f"{'='*70}\n")


def plot_comparisons(result1: Dict, result2: Dict, output_dir: str):
    """8개 비교 시각화 생성"""
    output_path = Path(output_dir) / "plots"
    output_path.mkdir(parents=True, exist_ok=True)

    pv1 = result1['portfolio_values']
    pv2 = result2['portfolio_values']
    name1 = result1['model_name'].upper()
    name2 = result2['model_name'].upper()

    # 1. Portfolio Value Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(pv1, label=name1, linewidth=2)
    plt.plot(pv2, label=name2, linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "portfolio_value_comparison.png", dpi=150)
    plt.close()

    # 2. Returns Distribution
    returns1 = (pv1[1:] - pv1[:-1]) / pv1[:-1]
    returns2 = (pv2[1:] - pv2[:-1]) / pv2[:-1]

    plt.figure(figsize=(12, 6))
    plt.hist(returns1, bins='auto', alpha=0.5, label=f'{name1} (μ={returns1.mean():.4f})', edgecolor='black')
    plt.hist(returns2, bins='auto', alpha=0.5, label=f'{name2} (μ={returns2.mean():.4f})', edgecolor='black')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.title('Returns Distribution Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "returns_distribution.png", dpi=150)
    plt.close()

    # 3. Drawdown Comparison
    def calculate_drawdown(pv):
        cummax = np.maximum.accumulate(pv)
        dd = (pv - cummax) / cummax
        return dd

    dd1 = calculate_drawdown(pv1)
    dd2 = calculate_drawdown(pv2)

    plt.figure(figsize=(12, 6))
    plt.fill_between(range(len(dd1)), dd1, 0, alpha=0.3, label=name1)
    plt.fill_between(range(len(dd2)), dd2, 0, alpha=0.3, label=name2)
    plt.plot(dd1, linewidth=1)
    plt.plot(dd2, linewidth=1)
    plt.xlabel('Step')
    plt.ylabel('Drawdown')
    plt.title('Drawdown Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "drawdown_comparison.png", dpi=150)
    plt.close()

    # 4. Performance Metrics Bar Chart
    m1 = result1['metrics']
    m2 = result2['metrics']

    metrics = ['Sharpe\nRatio', 'Sortino\nRatio', 'Calmar\nRatio']
    values1 = [m1['sharpe_ratio'], m1['sortino_ratio'], m1['calmar_ratio']]
    values2 = [m2['sharpe_ratio'], m2['sortino_ratio'], m2['calmar_ratio']]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, values1, width, label=name1, alpha=0.8)
    ax.bar(x + width/2, values2, width, label=name2, alpha=0.8)

    ax.set_ylabel('Value')
    ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path / "performance_metrics.png", dpi=150)
    plt.close()

    # 5. Risk Metrics Bar Chart
    metrics = ['Max Drawdown\n(%)', 'Volatility\n(%)', 'VaR 95%\n(%)', 'CVaR 95%\n(%)']
    values1 = [abs(m1['max_drawdown'])*100, m1['volatility']*100, abs(m1['var_95'])*100, abs(m1['cvar_95'])*100]
    values2 = [abs(m2['max_drawdown'])*100, m2['volatility']*100, abs(m2['var_95'])*100, abs(m2['cvar_95'])*100]

    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, values1, width, label=name1, alpha=0.8)
    ax.bar(x + width/2, values2, width, label=name2, alpha=0.8)

    ax.set_ylabel('Value (%)')
    ax.set_title('Risk Metrics Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path / "risk_metrics.png", dpi=150)
    plt.close()

    # 6. Cumulative Returns
    cumret1 = (pv1 / pv1[0] - 1) * 100
    cumret2 = (pv2 / pv2[0] - 1) * 100

    plt.figure(figsize=(12, 6))
    plt.plot(cumret1, label=name1, linewidth=2)
    plt.plot(cumret2, label=name2, linewidth=2)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Step')
    plt.ylabel('Cumulative Return (%)')
    plt.title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "cumulative_returns.png", dpi=150)
    plt.close()

    # 7. Rolling Sharpe Ratio (30-day window)
    def rolling_sharpe(returns, window=30, risk_free=0.02/252):
        rolling_mean = pd.Series(returns).rolling(window).mean()
        rolling_std = pd.Series(returns).rolling(window).std()
        return (rolling_mean - risk_free) / rolling_std * np.sqrt(252)

    sharpe1 = rolling_sharpe(returns1)
    sharpe2 = rolling_sharpe(returns2)

    plt.figure(figsize=(12, 6))
    plt.plot(sharpe1, label=name1, linewidth=1.5, alpha=0.8)
    plt.plot(sharpe2, label=name2, linewidth=1.5, alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Step')
    plt.ylabel('Rolling Sharpe Ratio (30-day)')
    plt.title('Rolling Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "rolling_sharpe.png", dpi=150)
    plt.close()

    # 8. Crisis Response (IRT only)
    irt_result = None
    for result in [result1, result2]:
        if result['irt_data'] is not None:
            irt_result = result
            break

    if irt_result is not None:
        irt_data = irt_result['irt_data']
        crisis_levels = irt_data['crisis_levels']
        pv_irt = irt_result['portfolio_values'][1:]  # IRT 모델의 포트폴리오

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Crisis levels
        ax1.fill_between(range(len(crisis_levels)), crisis_levels, alpha=0.3, label='Crisis Level')
        ax1.plot(crisis_levels, linewidth=1.5)
        ax1.set_ylabel('Crisis Level')
        ax1.set_title('T-Cell Crisis Detection', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Portfolio returns during crisis
        ret_irt = (pv_irt[1:] - pv_irt[:-1]) / pv_irt[:-1]
        crisis_mask = crisis_levels[1:] > 0.5  # crisis_levels의 길이에 맞게 조정

        # crisis_mask의 길이를 ret_irt에 맞게 조정
        min_len = min(len(ret_irt), len(crisis_mask))
        ret_irt = ret_irt[:min_len]
        crisis_mask = crisis_mask[:min_len]

        ax2.scatter(np.where(~crisis_mask)[0], ret_irt[~crisis_mask],
                   alpha=0.3, s=10, label='Normal', color='blue')
        ax2.scatter(np.where(crisis_mask)[0], ret_irt[crisis_mask],
                   alpha=0.5, s=20, label='Crisis', color='red')
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Daily Return')
        ax2.set_title(f'{irt_result["model_name"].upper()} Returns (Crisis vs Normal)',
                     fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "crisis_response.png", dpi=150)
        plt.close()

    print(f"\n시각화 8개 생성 완료: {output_path}")


def save_results(result1: Dict, result2: Dict, output_dir: str):
    """결과를 JSON과 텍스트로 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # JSON 저장
    summary = {
        'model1': {
            'name': result1['model_name'],
            'path': result1['model_path'],
            'metrics': result1['metrics']
        },
        'model2': {
            'name': result2['model_name'],
            'path': result2['model_path'],
            'metrics': result2['metrics']
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(output_path / "comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"결과 저장 완료: {output_path / 'comparison_summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="두 모델 비교 분석")
    parser.add_argument("--model1", type=str, required=True,
                       help="첫 번째 모델 경로 (디렉토리 또는 파일)")
    parser.add_argument("--model2", type=str, required=True,
                       help="두 번째 모델 경로 (디렉토리 또는 파일)")
    parser.add_argument("--use-best", action="store_true",
                       help="Use best_model instead of final model (default: False)")
    parser.add_argument("--output", type=str, default="comparison_results",
                       help="출력 디렉토리 (기본: comparison_results)")
    parser.add_argument("--test-start", type=str, default=TEST_START_DATE,
                       help=f"테스트 시작 날짜 (기본: {TEST_START_DATE})")
    parser.add_argument("--test-end", type=str, default=TEST_END_DATE,
                       help=f"테스트 종료 날짜 (기본: {TEST_END_DATE})")

    args = parser.parse_args()

    # 모델 경로 해석
    print(f"\n{'='*70}")
    print(f"Resolving model paths...")
    print(f"{'='*70}")

    model_path1 = resolve_model_path(args.model1, args.use_best)
    print(f"Model 1: {model_path1}")

    model_path2 = resolve_model_path(args.model2, args.use_best)
    print(f"Model 2: {model_path2}")

    # 1. 모델 1 평가
    result1 = evaluate_model(model_path1, args.test_start, args.test_end)

    # 2. 모델 2 평가
    result2 = evaluate_model(model_path2, args.test_start, args.test_end)

    # 3. 비교 테이블 출력
    print_comparison_table(result1, result2)

    # 4. 시각화 생성
    plot_comparisons(result1, result2, args.output)

    # 5. 결과 저장
    save_results(result1, result2, args.output)

    print(f"\n{'='*70}")
    print(f"All tasks completed!")
    print(f"  Output: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
