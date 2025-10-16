# scripts/train.py
# 여러 강화학습 알고리즘을 학습·평가하는 공통 실행 스크립트를 정의한다.

"""
통합 강화학습 학습 스크립트

`config.py`에 정의된 하이퍼파라미터를 바탕으로 Stable Baselines3 알고리즘을 학습하고,
학습한 모델을 평가하거나 저장하는 전체 파이프라인을 제공한다.

사용 예:
    # SAC 학습 후 평가까지 수행
    python scripts/train.py --model sac --mode both

    # PPO 모델을 학습만 진행
    python scripts/train.py --model ppo --mode train --episodes 100

    # 저장된 SAC 모델을 평가
    python scripts/train.py --model sac --mode test --checkpoint logs/sac/20251004_120000/best_model.zip
"""

import argparse
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd

from finrl.config import (
    INDICATORS,
    SAC_PARAMS,
    PPO_PARAMS,
    A2C_PARAMS,
    TD3_PARAMS,
    DDPG_PARAMS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
)
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.agents.stablebaselines3 import StrictEvalCallback
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import SAC, PPO, A2C, TD3, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# 평가 함수 로드를 위해 현재 디렉터리를 우선 추가한다.
import sys

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import evaluate_model


# 지원 알고리즘명과 Stable Baselines3 클래스 매핑
MODEL_CLASSES = {"sac": SAC, "ppo": PPO, "a2c": A2C, "td3": TD3, "ddpg": DDPG}

# 알고리즘별 기본 하이퍼파라미터
MODEL_PARAMS = {
    "sac": SAC_PARAMS,
    "ppo": PPO_PARAMS,
    "a2c": A2C_PARAMS,
    "td3": TD3_PARAMS,
    "ddpg": DDPG_PARAMS,
}


def save_metadata(output_dir, tickers, train_start, train_end, test_start, test_end):
    """학습 시 사용한 종목·기간 정보를 JSON으로 저장해 평가 단계에서 재사용한다."""
    metadata = {
        "tickers": tickers,
        "train_period": {"start": train_start, "end": train_end},
        "test_period": {"start": test_start, "end": test_end},
        "n_stocks": len(tickers),
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata saved to: {metadata_path}")


def load_metadata(model_dir):
    """저장된 메타데이터를 로드하여 평가 시 동일한 종목 구성을 복원한다."""
    metadata_path = os.path.join(model_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"메타데이터 파일이 없습니다: {metadata_path}\n"
            f"학습 시 사용된 주식 목록을 알 수 없어 평가를 진행할 수 없습니다."
        )

    with open(metadata_path, "r") as f:
        return json.load(f)


def create_env(df, stock_dim, tech_indicators):
    """주어진 데이터프레임을 기반으로 FinRL 주식 거래 환경을 생성한다."""

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
        "print_verbosity": 500,
    }

    return StockTradingEnv(**env_kwargs)


def preprocess_params(model_name, params):
    """config.py에서 불러온 하이퍼파라미터를 Stable Baselines3 형식에 맞게 정제한다."""

    processed = params.copy()

    # SAC의 ent_coef 처리
    if model_name == "sac" and "ent_coef" in processed:
        if isinstance(processed["ent_coef"], str):
            if processed["ent_coef"].startswith("auto"):
                # "auto_0.1" -> "auto"로 변환 (SB3는 초기값 지정 불가)
                processed["ent_coef"] = "auto"

    return processed


def train_model(args):
    """입력된 인자를 바탕으로 선택한 알고리즘을 학습하고 결과물을 저장한다."""

    print("=" * 70)
    print(f"{args.model.upper()} Training - Dow Jones 30")
    print("=" * 70)

    # 출력 디렉토리
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output, args.model, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    print("\n[Config]")
    print(f"  Model: {args.model.upper()}")
    print(f"  Stocks: Dow Jones 30 ({len(DOW_30_TICKER)} tickers)")
    print(f"  Train period: {args.train_start} ~ {args.train_end}")
    print(f"  Test period: {args.test_start} ~ {args.test_end}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Output directory: {log_dir}")

    # 1. 데이터 다운로드
    print("\n[1/5] Downloading data...")
    df = YahooDownloader(
        start_date=args.train_start, end_date=args.test_end, ticker_list=DOW_30_TICKER
    ).fetch_data()
    print(f"  Downloaded: {df.shape[0]} rows")

    # 2. Feature Engineering
    print("\n[2/5] Running feature engineering...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        user_defined_feature=False,
    )
    df_processed = fe.preprocess_data(df)
    print(f"  Feature columns: {df_processed.shape[1]}")

    # 3. Train/Test Split
    print("\n[3/5] Splitting train/test data...")
    train_df = data_split(df_processed, args.train_start, args.train_end)
    test_df = data_split(df_processed, args.test_start, args.test_end)
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")

    # 4. 환경 생성
    print("\n[4/5] Creating environments...")
    actual_tickers = train_df.tic.unique().tolist()
    stock_dim = len(actual_tickers)

    # 데이터 누락 경고
    if stock_dim != len(DOW_30_TICKER):
        removed_count = len(DOW_30_TICKER) - stock_dim
        excluded_tickers = set(DOW_30_TICKER) - set(actual_tickers)
        print(f"  ⚠️ Warning: {removed_count} tickers were excluded due to missing data")
        print(f"      Excluded tickers: {sorted(excluded_tickers)}")

    print(f"  Effective stock count: {stock_dim}")
    print(f"  Sample tickers: {actual_tickers[:5]}... (truncated)")
    train_env = create_env(train_df, stock_dim, INDICATORS)
    test_env = create_env(test_df, stock_dim, INDICATORS)
    print(f"  State space: {train_env.state_space}")
    print(f"  Action space: {train_env.action_space}")

    # 5. 모델 학습
    print(f"\n[5/5] Training {args.model.upper()}...")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(log_dir, "checkpoints"),
        name_prefix=f"{args.model}_model",
    )

    # Monitor wrapper로 평가 환경 래핑
    eval_env = Monitor(test_env)

    eval_callback = StrictEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    # 모델 파라미터 준비
    model_class = MODEL_CLASSES[args.model]
    model_params = preprocess_params(args.model, MODEL_PARAMS[args.model])

    print(f"  Hyperparameters: {model_params}")

    # 모델 생성
    model = model_class(
        "MlpPolicy",
        train_env,
        **model_params,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
    )

    # 총 timesteps 계산
    total_timesteps = 250 * args.episodes
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Starting training loop...")

    # 학습
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # 최종 모델 저장
    final_model_path = os.path.join(log_dir, f"{args.model}_final.zip")
    model.save(final_model_path)

    # 메타데이터 저장 (평가 시 일관성 유지)
    save_metadata(
        output_dir=log_dir,
        tickers=actual_tickers,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"  Final model: {final_model_path}")
    print(f"  Best model: {os.path.join(log_dir, 'best_model', 'best_model.zip')}")
    print(f"  Logs: {log_dir}")
    print(f"  Metadata: {os.path.join(log_dir, 'metadata.json')}")

    return log_dir, final_model_path


def test_model(args, model_path=None):
    """학습된 모델을 불러와 동일한 종목 구성을 기준으로 테스트 구간을 평가한다."""

    print("=" * 70)
    print(f"{args.model.upper()} Evaluation")
    print("=" * 70)

    # 모델 경로 결정
    if model_path is None:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --mode test is used")
        model_path = args.checkpoint

    # 메타데이터 로드 (학습 시 사용된 ticker 정보)
    model_dir = os.path.dirname(model_path)
    metadata = load_metadata(model_dir)
    train_tickers = metadata["tickers"]

    print("\n[Config]")
    print(f"  Model path: {model_path}")
    print(f"  Test period: {args.test_start} ~ {args.test_end}")

    print("\n[Metadata]")
    print(f"  Training tickers: {len(train_tickers)}")
    print(f"  Sample tickers: {train_tickers[:5]}... (truncated)")

    # evaluate.py의 evaluate_model() 재사용
    portfolio_values, exec_returns, value_returns, irt_data, metrics, _artefacts = (
        evaluate_model(
            model_path=model_path,
            model_class=MODEL_CLASSES[args.model],
            test_start=args.test_start,
            test_end=args.test_end,
            stock_tickers=train_tickers,  # 학습 시 사용된 ticker만
            tech_indicators=INDICATORS,
            initial_amount=1000000,
            verbose=True,
        )
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

    return {
        "portfolio_values": portfolio_values,
        "final_value": final_value,
        "total_return": total_return,
        "metrics": metrics,
        "steps": step,
    }


def main():
    parser = argparse.ArgumentParser(
        description="통합 RL 학습/평가 파이프라인"
    )

    # model and mode
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["sac", "ppo", "a2c", "td3", "ddpg"],
        help="학습/평가할 RL 알고리즘",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="실행 모드 (기본: both)",
    )

    # data periods (defaults defined in config.py)
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
        help=f"평가 시작일 (기본: {TEST_START_DATE})",
    )
    parser.add_argument(
        "--test-end",
        type=str,
        default=TEST_END_DATE,
        help=f"평가 종료일 (기본: {TEST_END_DATE})",
    )

    # 학습 설정
    parser.add_argument(
        "--episodes", type=int, default=200, help="학습 에피소드 수 (기본값: 200)"
    )
    parser.add_argument(
        "--output", type=str, default="logs", help="출력 디렉터리 (기본값: logs)"
    )

    # 평가 전용
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="평가 시 사용할 체크포인트 경로 (--mode test에 필수)",
    )

    args = parser.parse_args()

    # 실행
    if args.mode == "train":
        log_dir, model_path = train_model(args)
        print(f"\nTraining complete. Results directory: {log_dir}")

    elif args.mode == "test":
        results = test_model(args)
        print(
            f"\nEvaluation complete. Final return: {results['total_return']*100:.2f}%"
        )

    elif args.mode == "both":
        log_dir, model_path = train_model(args)
        print(
            f"\nTraining finished. Starting evaluation with the same configuration...\n"
        )
        results = test_model(args, model_path=model_path)
        print(f"\nFull pipeline completed.")
        print(f"  Training outputs: {log_dir}")
        print(f"  Final return: {results['total_return']*100:.2f}%")


if __name__ == "__main__":
    main()
