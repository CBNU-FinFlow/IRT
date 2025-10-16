# tests/test_finrl_minimal.py

"""
FinRL 최소 실행 테스트

목적:
1. FinRL 환경 정상 작동 확인
2. SAC baseline 10k steps 학습 테스트
3. 기본 데이터 다운로드 및 전처리 검증

사용법:
    python tests/test_finrl_minimal.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

def test_minimal():
    """최소 실행 테스트: 소수 종목, 짧은 기간"""

    print("=" * 60)
    print("Starting FinRL minimal smoke test")
    print("=" * 60)

    # 1. 데이터 다운로드 (소수 종목, 짧은 기간)
    print("\n[1/5] Downloading data...")
    ticker_list = DOW_30_TICKER[:5]  # 5개 종목만
    start_date = '2023-01-01'
    end_date = '2024-01-01'

    df = YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=ticker_list
    ).fetch_data()

    print(f"  Downloaded: {df.shape[0]} rows, {len(ticker_list)} tickers")

    # 2. Feature Engineering
    print("\n[2/5] Running feature engineering...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS[:4],  # 4개 지표만
        use_turbulence=False,  # 짧은 기간이므로 turbulence 비활성화
        user_defined_feature=False
    )
    df_processed = fe.preprocess_data(df)
    print(f"  Feature engineering completed: {df_processed.shape[1]} features")

    # 3. Train/Test Split
    print("\n[3/5] Splitting train/test data...")
    train_df = data_split(df_processed, start_date, '2023-10-01')

    print(f"  Train samples: {len(train_df)}")

    # 4. 환경 생성
    print("\n[4/5] Creating StockTradingEnv...")

    stock_dim = len(ticker_list)

    # state_space 계산
    # state = [cash] + [prices] * stock_dim + [holdings] * stock_dim + [indicators] * stock_dim
    # indicators = 4개 (macd, boll_ub, boll_lb, rsi_30)
    state_space = 1 + (len(INDICATORS[:4]) + 2) * stock_dim

    env_kwargs = {
        "df": train_df,
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": INDICATORS[:4],
        "print_verbosity": 10
    }

    env = StockTradingEnv(**env_kwargs)
    print("  Environment created")
    print(f"    State space: {state_space}")
    print(f"    Action space: {stock_dim}")

    # 5. SAC 학습 (짧게)
    print("\n[5/5] Training SAC for 10k steps...")

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=64,
        learning_starts=100
    )

    model.learn(total_timesteps=10000, progress_bar=True)

    print("\n" + "=" * 60)
    print("✅ FinRL minimal smoke test succeeded!")
    print("=" * 60)

    # 간단한 평가
    print("\n[Bonus] Evaluating the trained model...")
    obs, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        done = done or truncated

    # state를 numpy array로 변환
    state = np.array(env.state)
    cash = state[0]
    prices = state[1:stock_dim+1]
    holdings = state[stock_dim+1:2*stock_dim+1]
    portfolio_value = cash + np.sum(prices * holdings)

    print(f"  Final portfolio value: ${portfolio_value:.2f}")
    print(f"  Total reward: {total_reward:.4f}")

    return True

if __name__ == "__main__":
    success = test_minimal()
    if not success:
        exit(1)
