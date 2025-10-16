# scripts/train_finrl_standard.py

"""
FinRL 표준 파이프라인 학습 스크립트

DRLAgent를 사용해 FinRL 논문과 동일한 설정으로 베이스라인 모델을 학습·평가합니다.

사용 예:
    # SAC 학습 및 평가 (FinRL 표준)
    python scripts/train_finrl_standard.py --model sac --mode both

    # PPO 학습만 수행
    python scripts/train_finrl_standard.py --model ppo --mode train --timesteps 100000

    # 저장된 모델 평가
    python scripts/train_finrl_standard.py --model sac --mode test --checkpoint trained_models/sac_50k.zip
"""

import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd

from finrl.config import (
    INDICATORS,
    DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR,
    TRAIN_START_DATE, TRAIN_END_DATE,
    TEST_START_DATE, TEST_END_DATE
)
from finrl.config_tickers import DOW_30_TICKER
from finrl.main import check_and_make_directories
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure


def train_model(args):
    """DRLAgent를 사용한 표준 학습 파이프라인"""

    print("=" * 70)
    print(f"{args.model.upper()} Training (FinRL standard pipeline)")
    print("=" * 70)

    # 출력 디렉토리 (logs/ 아래에 통일)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output, f"finrl_{args.model}", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # Config 경로를 log_dir 하위로 동적 override (타임스탬프 통합)
    import finrl.config as config
    config.DATA_SAVE_DIR = os.path.join(log_dir, "datasets")
    config.TRAINED_MODEL_DIR = os.path.join(log_dir, "trained_models")
    config.TENSORBOARD_LOG_DIR = os.path.join(log_dir, "tensorboard_log")
    config.RESULTS_DIR = os.path.join(log_dir, "results")

    # 디렉토리 생성 (override된 경로 사용)
    check_and_make_directories([
        config.DATA_SAVE_DIR,
        config.TRAINED_MODEL_DIR,
        config.TENSORBOARD_LOG_DIR,
        config.RESULTS_DIR
    ])

    print("\n[Training Config]")
    print(f"  Model: {args.model.upper()}")
    print(f"  Stocks: Dow Jones 30 ({len(DOW_30_TICKER)} tickers)")
    print(f"  Train period: {args.train_start} ~ {args.train_end}")
    print(f"  Test period: {args.test_start} ~ {args.test_end}")
    print(f"  Total timesteps: {args.timesteps}")
    print(f"  Output directory: {log_dir}")

    # 1. 데이터 다운로드
    print("\n[1/6] Downloading data...")
    df = YahooDownloader(
        start_date=args.train_start,
        end_date=args.test_end,
        ticker_list=DOW_30_TICKER
    ).fetch_data()
    print(f"  Downloaded: {df.shape[0]} rows")

    # 2. Feature Engineering
    print("\n[2/6] Running feature engineering...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        user_defined_feature=False
    )
    processed = fe.preprocess_data(df)
    print(f"  Feature columns: {processed.shape[1]}")

    # 3. Train/Test Split
    print("\n[3/6] Splitting train/test data...")
    train_df = data_split(processed, args.train_start, args.train_end)
    test_df = data_split(processed, args.test_start, args.test_end)
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")

    # 4. 환경 생성
    print("\n[4/6] Building training environment...")
    stock_dim = len(train_df.tic.unique())

    # 데이터 누락 경고
    if stock_dim != len(DOW_30_TICKER):
        removed_count = len(DOW_30_TICKER) - stock_dim
        print(f"  ⚠️ Warning: {removed_count} tickers excluded due to missing data")
        print(f"      (e.g., incomplete history for early-2008 listings such as Visa (V))")

    print(f"  Effective stock count: {stock_dim}")
    state_space = 1 + (len(INDICATORS) + 2) * stock_dim

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
        "tech_indicator_list": INDICATORS,
        "print_verbosity": 500
    }

    e_train_gym = StockTradingEnv(**env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()  # FinRL 표준: get_sb_env() 사용

    print(f"  State space: {state_space}")
    print(f"  Action space: {stock_dim}")
    print(f"  Environment type: {type(env_train)}")

    print(f"\n[5/6] Training {args.model.upper()} with DRLAgent...")

    agent = DRLAgent(env=env_train)

    # config.py의 MODEL_KWARGS 자동 로드 (model_kwargs=None이면 자동)
    model = agent.get_model(
        args.model,
        policy="MlpPolicy",
        model_kwargs=None,  # None이면 config.py 사용
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tensorboard")
    )

    print(f"  Model class: {type(model)}")
    print(f"  Total timesteps: {args.timesteps}")

    # Logger 설정 (log_dir 아래)
    log_path = os.path.join(log_dir, "logs")
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # 학습 (callbacks=[] → TensorboardCallback 비활성화, off-policy 알고리즘 호환)
    trained_model = agent.train_model(
        model=model,
        tb_log_name=args.model,
        total_timesteps=args.timesteps,
        callbacks=[]  # TensorboardCallback 비활성화 (SAC/TD3/DDPG는 rollout_buffer 없음)
    )

    # 모델 저장 (log_dir 아래)
    model_path = os.path.join(log_dir, f"{args.model}_{args.timesteps//1000}k.zip")
    trained_model.save(model_path)

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"  Saved model: {model_path}")
    print(f"  Logs: {log_dir}")

    return trained_model, model_path, log_dir


def test_model(args, trained_model=None, model_path=None, log_dir=None):
    """DRLAgent.DRL_prediction()을 사용한 평가"""

    print("=" * 70)
    print(f"{args.model.upper()} Evaluation (FinRL standard pipeline)")
    print("=" * 70)

    # log_dir 결정
    if log_dir is None:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --mode test is used")
        # use checkpoint directory
        log_dir = os.path.dirname(args.checkpoint)
        print(f"\n  Using checkpoint directory for output: {log_dir}")

    print("\n[Evaluation Config]")
    print(f"  Test period: {args.test_start} ~ {args.test_end}")
    print(f"  Output directory: {log_dir}")

    # 1. Data preparation
    print("\n[1/4] Downloading evaluation data...")
    df = YahooDownloader(
        start_date=args.test_start,
        end_date=args.test_end,
        ticker_list=DOW_30_TICKER
    ).fetch_data()
    print(f"  Downloaded: {df.shape[0]} rows")

    # 2. Feature engineering
    print("\n[2/4] Running feature engineering...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        user_defined_feature=False
    )
    processed = fe.preprocess_data(df)
    test_df = data_split(processed, args.test_start, args.test_end)
    print(f"  Test samples: {len(test_df)}")

    # 3. Create test environment
    print("\n[3/4] Creating test environment...")
    stock_dim = len(DOW_30_TICKER)
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
        "print_verbosity": 500
    }

    e_test_gym = StockTradingEnv(**env_kwargs)

    # 4. Load model and run evaluation
    print("\n[4/4] Running DRLAgent.DRL_prediction()...")

    if trained_model is None:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --mode test is used")
        # 모델 로드
        from stable_baselines3 import SAC, PPO, A2C, TD3, DDPG
        MODEL_CLASSES = {'sac': SAC, 'ppo': PPO, 'a2c': A2C, 'td3': TD3, 'ddpg': DDPG}
        model = MODEL_CLASSES[args.model].load(args.checkpoint)
        print(f"  Model loaded from: {args.checkpoint}")
    else:
        model = trained_model
        print("  Using the trained model instance from this session")

    # DRLAgent.DRL_prediction() 사용 (FinRL 표준)
    account_memory, actions_memory = DRLAgent.DRL_prediction(
        model=model,
        environment=e_test_gym,
        deterministic=True
    )

    # 5. 결과 저장 (log_dir 아래)
    account_path = os.path.join(log_dir, "account_value_test.csv")
    actions_path = os.path.join(log_dir, "actions_test.csv")

    account_memory.to_csv(account_path, index=False)
    actions_memory.to_csv(actions_path, index=False)

    # 6. 결과 출력
    initial_value = account_memory['account_value'].iloc[0]
    final_value = account_memory['account_value'].iloc[-1]
    total_return = (final_value - initial_value) / initial_value

    # Daily return 계산
    account_memory['daily_return'] = account_memory['account_value'].pct_change(1)
    sharpe_ratio = (
        (252 ** 0.5) * account_memory['daily_return'].mean() / account_memory['daily_return'].std()
        if account_memory['daily_return'].std() > 0 else 0
    )

    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print("\n[Period]")
    print(f"  Start: {args.test_start}")
    print(f"  End: {args.test_end}")
    print(f"  Steps: {len(account_memory)}")

    print("\n[Returns]")
    print(f"  Initial value: ${initial_value:,.2f}")
    print(f"  Final value: ${final_value:,.2f}")
    print(f"  Total return: {total_return*100:.2f}%")
    print(f"  Sharpe ratio: {sharpe_ratio:.3f}")

    print("\n[Saved files]")
    print(f"  Account values: {account_path}")
    print(f"  Actions: {actions_path}")

    print("\n" + "=" * 70)

    return {
        'account_memory': account_memory,
        'actions_memory': actions_memory,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio
    }


def main():
    parser = argparse.ArgumentParser(description="FinRL 표준 파이프라인 학습/평가")

    # 모델 및 모드
    parser.add_argument("--model", type=str, required=True,
                        choices=['sac', 'ppo', 'a2c', 'td3', 'ddpg'],
                        help="RL 알고리즘 선택")
    parser.add_argument("--mode", type=str, default="both",
                        choices=['train', 'test', 'both'],
                        help="실행 모드 (기본: both)")

    # 데이터 기간 (config.py 기본값)
    parser.add_argument("--train-start", type=str, default=TRAIN_START_DATE,
                        help=f"훈련 시작일 (기본: {TRAIN_START_DATE})")
    parser.add_argument("--train-end", type=str, default=TRAIN_END_DATE,
                        help=f"훈련 종료일 (기본: {TRAIN_END_DATE})")
    parser.add_argument("--test-start", type=str, default=TEST_START_DATE,
                        help=f"테스트 시작일 (기본: {TEST_START_DATE})")
    parser.add_argument("--test-end", type=str, default=TEST_END_DATE,
                        help=f"테스트 종료일 (기본: {TEST_END_DATE})")

    # 학습 설정
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="총 타임스텝 수 (기본: 50000, FinRL 표준)")
    parser.add_argument("--output", type=str, default="logs",
                        help="출력 디렉터리 (기본: logs)")

    # 평가 전용
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="평가용 체크포인트 경로 (--mode test에 필수)")

    args = parser.parse_args()

    # execution
    if args.mode == "train":
        trained_model, model_path, log_dir = train_model(args)
        print(f"\nTraining complete. Results directory: {log_dir}")

    elif args.mode == "test":
        results = test_model(args)
        print(f"\nEvaluation complete. Sharpe ratio: {results['sharpe_ratio']:.3f}")

    elif args.mode == "both":
        trained_model, model_path, log_dir = train_model(args)
        print(f"\nTraining finished. Starting evaluation...\n")
        results = test_model(args, trained_model=trained_model, model_path=model_path, log_dir=log_dir)
        print(f"\nFull pipeline completed.")
        print(f"  Outputs: {log_dir}")
        print(f"  Sharpe ratio: {results['sharpe_ratio']:.3f}")


if __name__ == "__main__":
    main()
