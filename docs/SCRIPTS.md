# Scripts 사용 가이드

FinRL-IRT 프로젝트의 학습 및 평가 스크립트 사용법을 설명한다.

---

## 목차

1. [개요](#개요)
2. [scripts/train.py](#scriptstrain.py) - 일반 RL 알고리즘
3. [scripts/train_irt.py](#scriptstrain_irt.py) - IRT Policy 학습
4. [scripts/train_finrl_standard.py](#scriptstrain_finrl_standard.py) - FinRL 표준
5. [scripts/evaluate.py](#scriptsevaluate.py) - 평가
6. [스크립트 비교](#스크립트-비교)
7. [저장 위치](#저장-위치)
8. [사용 시나리오](#사용-시나리오)

---

## 개요

본 프로젝트는 세 가지 학습 파이프라인을 제공한다:

| 스크립트                  | 목적                  | 파이프라인      | 특징                          |
| ------------------------- | --------------------- | --------------- | ----------------------------- |
| `train.py`                | 일반 RL 알고리즘 학습 | SB3 직접 사용   | SAC, PPO, A2C, TD3, DDPG 지원 |
| `train_irt.py`            | IRT Policy 학습       | SAC + IRTPolicy | 위기 적응형 포트폴리오 관리   |
| `train_finrl_standard.py` | FinRL 표준 베이스라인 | DRLAgent 사용   | 논문 재현성 검증              |

모든 스크립트는 결과를 `logs/` 아래에 타임스탬프로 저장한다.

---

## scripts/train.py

IRT Custom Policy와 통합 용이한 학습 스크립트. Stable Baselines3를 직접 사용하여 유연성을 확보한다.

### 사용법

```bash
python scripts/train.py [OPTIONS]
```

### 주요 인자

| 인자            | 타입 | 기본값       | 설명                                             |
| --------------- | ---- | ------------ | ------------------------------------------------ |
| `--model`       | str  | **필수**     | RL 알고리즘 (`sac`, `ppo`, `a2c`, `td3`, `ddpg`) |
| `--mode`        | str  | `both`       | 실행 모드 (`train`, `test`, `both`)              |
| `--episodes`    | int  | 200          | 에피소드 수 (총 timesteps = 250 × episodes)      |
| `--train-start` | str  | `2008-01-01` | 학습 시작일                                      |
| `--train-end`   | str  | `2020-12-31` | 학습 종료일                                      |
| `--test-start`  | str  | `2021-01-01` | 테스트 시작일                                    |
| `--test-end`    | str  | `2024-12-31` | 테스트 종료일                                    |
| `--output`      | str  | `logs`       | 출력 디렉토리                                    |
| `--checkpoint`  | str  | `None`       | 평가 전용 모드에서 모델 경로                     |

### 예시

**1. SAC 학습 및 평가 (200 episodes)**

```bash
python scripts/train.py --model sac --mode both --episodes 200
```

출력:

```
logs/sac/20251004_120000/
├── checkpoints/
│   ├── sac_model_10000_steps.zip
│   └── ...
├── best_model/
│   └── best_model.zip
├── sac_final.zip
├── metadata.json          # 학습 시 사용된 ticker 및 기간 정보
├── tensorboard/
└── eval/
```

**2. PPO 학습만 (커스텀 기간)**

```bash
python scripts/train.py \
  --model ppo \
  --mode train \
  --episodes 100 \
  --train-start 2010-01-01 \
  --train-end 2022-12-31
```

**3. 저장된 모델 평가만**

```bash
python scripts/train.py \
  --model sac \
  --mode test \
  --checkpoint logs/sac/20251004_120000/sac_final.zip
```

### 특징

- **config.py 사용**: `SAC_PARAMS`, `PPO_PARAMS` 등 자동 로드
- **Callbacks**: CheckpointCallback (10,000 steps마다), EvalCallback (5,000 steps마다)
- **TensorBoard**: `logs/{model}/{timestamp}/tensorboard/` 에서 확인
- **평가**: `model.predict()` 직접 호출 → portfolio value 수동 계산
- **Metadata 저장**: 학습 시 사용된 ticker list와 기간 정보를 `metadata.json`으로 저장
- **평가 일관성**: 평가 시 저장된 metadata를 로드하여 동일한 ticker로 데이터 필터링 (observation space 불일치 방지)

### 총 Timesteps 계산

- 에피소드당 약 250 steps (1년 거래일)
- 총 timesteps = 250 × `--episodes`
- 예: `--episodes 200` → 50,000 timesteps

---

## scripts/train_irt.py

IRT (Immune Replicator Transport) Policy를 학습하는 스크립트. SAC와 IRTPolicy를 조합하여 위기 적응형 포트폴리오 관리를 수행한다.

### 사용법

```bash
python scripts/train_irt.py [OPTIONS]
```

### 주요 인자

| 인자            | 타입 | 기본값       | 설명                                        |
| --------------- | ---- | ------------ | ------------------------------------------- |
| `--mode`        | str  | `both`       | 실행 모드 (`train`, `test`, `both`)         |
| `--episodes`    | int  | 200          | 에피소드 수 (총 timesteps = 250 × episodes) |
| `--train-start` | str  | `2008-01-01` | 학습 시작일                                 |
| `--train-end`   | str  | `2020-12-31` | 학습 종료일                                 |
| `--test-start`  | str  | `2021-01-01` | 테스트 시작일                               |
| `--test-end`    | str  | `2024-12-31` | 테스트 종료일                               |
| `--output`      | str  | `logs`       | 출력 디렉토리                               |
| `--checkpoint`  | str  | `None`       | 평가 전용 모드에서 모델 경로                |

### IRT 하이퍼파라미터

| 인자                   | 타입  | 기본값 | 설명                            |
| ---------------------- | ----- | ------ | ------------------------------- |
| `--emb-dim`            | int   | 128    | IRT embedding dimension         |
| `--m-tokens`           | int   | 6      | Epitope tokens 수               |
| `--M-proto`            | int   | 8      | Prototype 수                    |
| `--alpha`              | float | 0.3    | OT-Replicator mixing ratio      |
| `--eps`                | float | 0.10   | Sinkhorn entropy                |
| `--eta-0`              | float | 0.05   | Base learning rate (Replicator) |
| `--eta-1`              | float | 0.15   | Crisis increase (Replicator)    |
| `--market-feature-dim` | int   | 12     | Market feature dimension        |

### 예시

**1. IRT 학습 및 평가 (기본 설정)**

```bash
python scripts/train_irt.py --mode both
```

출력:

```
logs/irt/20251004_150000/
├── checkpoints/
│   ├── irt_model_10000_steps.zip
│   └── ...
├── best_model/
│   └── best_model.zip
├── irt_final.zip
├── tensorboard/
└── eval/
```

**2. 커스텀 alpha로 학습**

```bash
python scripts/train_irt.py \
  --mode train \
  --alpha 0.5 \
  --episodes 200 \
  --eps 0.15
```

**3. 저장된 IRT 모델 평가**

```bash
python scripts/train_irt.py \
  --mode test \
  --checkpoint logs/irt/20251004_150000/irt_final.zip
```

### 특징

- **SAC + IRTPolicy**: SAC 알고리즘에 IRT Custom Policy 적용
- **T-Cell**: 위기 감지 시스템 (crisis types, danger embedding, crisis level)
- **B-Cell IRT Actor**: Epitope encoder + Prototype decoders + IRT mixing
- **위기 적응**: T-Cell 출력에 따라 동적으로 포트폴리오 조정
- **config.py 사용**: `SAC_PARAMS` 자동 로드
- **Monitor wrapper**: Evaluation environment를 자동으로 Monitor로 감싸기 (경고 방지)

### IRT 수식

```
w_t = (1-α)·Replicator(w_{t-1}, f_t) + α·Transport(E_t, K, C_t)
```

- **Replicator**: 과거 가중치와 fitness 기반 진화
- **Transport**: Epitope → Prototype 최적 수송 (Sinkhorn)
- **alpha**: 두 메커니즘 혼합 비율 (0.3 = 70% Replicator, 30% OT)

### 성능 목표

- Sharpe Ratio: +10~15% vs Baseline
- Crisis MDD: -20~30% vs Baseline (위기 시 낮은 손실)
- Turnover: 적절한 수준 유지

---

## scripts/train_finrl_standard.py

FinRL 논문과 동일한 조건으로 베이스라인을 학습하는 스크립트. DRLAgent를 사용하여 표준 파이프라인을 준수한다.

### 사용법

```bash
python scripts/train_finrl_standard.py [OPTIONS]
```

### 주요 인자

| 인자            | 타입 | 기본값       | 설명                                             |
| --------------- | ---- | ------------ | ------------------------------------------------ |
| `--model`       | str  | **필수**     | RL 알고리즘 (`sac`, `ppo`, `a2c`, `td3`, `ddpg`) |
| `--mode`        | str  | `both`       | 실행 모드 (`train`, `test`, `both`)              |
| `--timesteps`   | int  | 50000        | 총 학습 timesteps (FinRL 표준)                   |
| `--train-start` | str  | `2008-01-01` | 학습 시작일                                      |
| `--train-end`   | str  | `2020-12-31` | 학습 종료일                                      |
| `--test-start`  | str  | `2021-01-01` | 테스트 시작일                                    |
| `--test-end`    | str  | `2024-12-31` | 테스트 종료일                                    |
| `--output`      | str  | `logs`       | 출력 디렉토리                                    |
| `--checkpoint`  | str  | `None`       | 평가 전용 모드에서 모델 경로                     |

### 예시

**1. SAC 학습 및 평가 (FinRL 표준, 50k timesteps)**

```bash
python scripts/train_finrl_standard.py --model sac --mode both
```

출력:

```
logs/finrl_sac/20251004_130000/
├── sac_50k.zip
├── account_value_test.csv
├── actions_test.csv
├── logs/
│   ├── progress.csv
│   └── events.out.tfevents.*
└── tensorboard/
```

**2. PPO 학습만 (100k timesteps)**

```bash
python scripts/train_finrl_standard.py \
  --model ppo \
  --mode train \
  --timesteps 100000
```

**3. 저장된 모델 평가만**

```bash
python scripts/train_finrl_standard.py \
  --model sac \
  --mode test \
  --checkpoint logs/finrl_sac/20251004_130000/sac_50k.zip
```

### 특징

- **DRLAgent 사용**: FinRL 표준 파이프라인
- **get_sb_env()**: DummyVecEnv 자동 래핑
- **TensorboardCallback**: `callbacks=[]`로 비활성화 (off-policy 알고리즘 호환)
  - SAC/TD3/DDPG는 `rollout_buffer` 없음 (`replay_buffer` 사용)
  - On-policy (PPO/A2C)만 `rollout_buffer` 있음
- **평가**: `DRLAgent.DRL_prediction()` → `account_memory` (DataFrame)
- **결과 형식**: FinRL 논문과 동일 (account_value, actions CSV)

### config.py 자동 로드

`model_kwargs=None`으로 설정하면 config.py의 하이퍼파라미터 자동 사용:

- `SAC_PARAMS = {"batch_size": 64, "buffer_size": 100000, ...}`
- `PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.01, ...}`
- 등

---

## scripts/evaluate.py

저장된 모델을 상세 평가하고 시각화하는 스크립트. 두 가지 평가 방식을 지원한다.

### 사용법

```bash
python scripts/evaluate.py [OPTIONS]
```

### 주요 인자

| 인자            | 타입 | 기본값        | 설명                                           |
| --------------- | ---- | ------------- | ---------------------------------------------- |
| `--model`       | str  | **필수**      | 모델 파일 경로 (`.zip`)                        |
| `--model-type`  | str  | 자동 감지     | 모델 타입 (`sac`, `ppo`, `a2c`, `td3`, `ddpg`) |
| `--method`      | str  | `direct`      | 평가 방식 (`direct`, `drlagent`)               |
| `--test-start`  | str  | `2021-01-01`  | 테스트 시작일                                  |
| `--test-end`    | str  | `2024-12-31`  | 테스트 종료일                                  |
| `--save-plot`   | flag | `False`       | 시각화 결과 저장                               |
| `--save-json`   | flag | `False`       | JSON 결과 저장                                 |
| `--output`      | str  | 모델 디렉토리 | Plot 출력 디렉토리                             |
| `--output-json` | str  | 모델 디렉토리 | JSON 출력 파일                                 |

### 평가 방식

**1. Direct 방식 (`--method direct`)**

- `train.py` 결과 평가용
- `model.predict()` 직접 호출
- portfolio value 수동 계산
- **주의**: `scripts/evaluate.py`는 아직 metadata 자동 로드를 지원하지 않음
  - `train.py --mode test`를 사용하면 metadata가 자동으로 로드됨
  - 또는 `evaluate.py` 수정 필요 (향후 개선 예정)

**2. DRLAgent 방식 (`--method drlagent`)**

- `train_finrl_standard.py` 결과 평가용
- `DRLAgent.DRL_prediction()` 사용
- `account_memory` (DataFrame) 반환

### 예시

**1. train.py 결과 평가 (Direct 방식)**

```bash
python scripts/evaluate.py \
  --model logs/sac/20251004_120000/sac_final.zip \
  --method direct \
  --save-plot \
  --save-json
```

**2. train_finrl_standard.py 결과 평가 (DRLAgent 방식)**

```bash
python scripts/evaluate.py \
  --model logs/finrl_sac/20251004_130000/sac_50k.zip \
  --method drlagent \
  --save-plot \
  --save-json
```

**3. 커스텀 테스트 기간**

```bash
python scripts/evaluate.py \
  --model logs/sac/20251004_120000/best_model/best_model.zip \
  --test-start 2024-01-01 \
  --test-end 2024-12-31 \
  --save-plot
```

### 출력 메트릭

```
Performance Metrics
======================================================================

[Period]
  Start: 2021-01-01
  End: 2024-12-31
  Steps: 1008

[Returns]
  Total Return: 45.23%
  Annualized Return: 12.87%

[Risk Metrics]
  Volatility (annualized): 18.42%
  Maximum Drawdown: -22.15%

[Risk-Adjusted Returns]
  Sharpe Ratio: 0.698
  Sortino Ratio: 1.023
  Calmar Ratio: 0.581

[Portfolio Value]
  Initial: $1,000,000.00
  Final: $1,452,300.00
  Profit/Loss: $452,300.00
```

### 시각화 (--save-plot)

생성되는 플롯:

1. **portfolio_value.png** - 포트폴리오 가치 추이
2. **drawdown.png** - Drawdown 차트
3. **returns_distribution.png** - 일별 수익률 분포

저장 위치: `{모델 디렉토리}/evaluation_plots/`

### JSON 출력 (--save-json)

```json
{
	"model_path": "logs/sac/20251004_120000/sac_final.zip",
	"model_type": "sac",
	"evaluation_method": "direct",
	"test_period": {
		"start": "2021-01-01",
		"end": "2024-12-31",
		"steps": 1008
	},
	"metrics": {
		"total_return": 0.4523,
		"annualized_return": 0.1287,
		"volatility": 0.1842,
		"sharpe_ratio": 0.698,
		"sortino_ratio": 1.023,
		"calmar_ratio": 0.581,
		"max_drawdown": -0.2215,
		"final_value": 1452300.0,
		"profit_loss": 452300.0
	},
	"timestamp": "2025-10-04T14:30:00.000000"
}
```

저장 위치: `{모델 디렉토리}/evaluation_results.json`

---

## 스크립트 비교

### 4개 스크립트 상세 비교

| 항목               | train.py                 | train_irt.py                      | train_finrl_standard.py  | evaluate.py                         |
| ------------------ | ------------------------ | --------------------------------- | ------------------------ | ----------------------------------- |
| **목적**           | 일반 RL 알고리즘 학습    | IRT Policy 학습                   | FinRL 표준 베이스라인    | 모델 평가                           |
| **알고리즘**       | SAC, PPO, A2C, TD3, DDPG | SAC + IRTPolicy                   | SAC, PPO, A2C, TD3, DDPG | -                                   |
| **파이프라인**     | SB3 직접 사용            | SB3 + IRTPolicy                   | DRLAgent + get_sb_env()  | Direct / DRLAgent                   |
| **환경 생성**      | `create_env()`           | `create_env()`                    | `get_sb_env()`           | `create_env()`                      |
| **DummyVecEnv**    | 자동 래핑 (SB3 내부)     | 자동 래핑 (SB3 내부)              | 수동 래핑 (get_sb_env)   | 평가 시 생성                        |
| **모델 생성**      | `MODEL_CLASSES[model]()` | `SAC(policy=IRTPolicy)`           | `DRLAgent.get_model()`   | 로드만                              |
| **하이퍼파라미터** | config.py (MODEL_PARAMS) | config.py (SAC_PARAMS) + IRT args | config.py (MODEL_KWARGS) | -                                   |
| **Metadata 저장**  | ✅ Yes (ticker, 기간)     | ✅ Yes (ticker, 기간)              | ❌ No                    | ❌ No (train.py 사용 권장)          |
| **TensorboardCallback** | CheckpointCallback, EvalCallback | CheckpointCallback, EvalCallback | `callbacks=[]` (비활성화) | -                          |
| **평가 방식**      | `model.predict()`        | `model.predict()`                 | `DRL_prediction()`       | `direct` / `drlagent`               |
| **FinRL 표준**     | ❌ No                    | ❌ No                             | ✅ Yes                   | drlagent만 Yes                      |
| **출력 위치**      | `logs/{model}/`          | `logs/irt/`                       | `logs/finrl_{model}/`    | 평가 결과                           |
| **시각화**         | -                        | -                                 | -                        | finrl/evaluation/visualizer.py 사용 |
| **IRT 플롯**       | -                        | 14개                              | -                        | 14개 (IRT Policy 전용)              |

### DummyVecEnv 래핑 방식

**1. 자동 래핑** (train.py, train_irt.py)

- StockTradingEnv를 직접 생성
- SB3 모델에 전달 시 내부적으로 DummyVecEnv 자동 래핑
- 코드가 간결하지만 FinRL 표준은 아님

```python
# train.py, train_irt.py
train_env = create_env(train_df, stock_dim, INDICATORS)  # StockTradingEnv
model = SAC("MlpPolicy", train_env, ...)  # SB3가 자동으로 DummyVecEnv 래핑
```

**2. 수동 래핑** (train_finrl_standard.py)

- `get_sb_env()` 메소드 사용
- StockTradingEnv를 DummyVecEnv로 명시적 래핑
- FinRL 표준 파이프라인 준수

```python
# train_finrl_standard.py
e_train_gym = StockTradingEnv(**env_kwargs)
env_train, _ = e_train_gym.get_sb_env()  # DummyVecEnv([lambda: self])
model = agent.get_model("sac", policy="MlpPolicy", ...)
```

**성능 차이**: 없음. 단순 코드 스타일 차이.

### 평가 방식 차이

| 방식         | 사용 스크립트           | 호출 메소드        | 반환값    | 특징                        |
| ------------ | ----------------------- | ------------------ | --------- | --------------------------- |
| **Direct**   | train.py, train_irt.py  | `model.predict()`  | action    | Portfolio value 수동 계산   |
| **DRLAgent** | train_finrl_standard.py | `DRL_prediction()` | DataFrame | FinRL 표준 (account_memory) |

**evaluate.py 방식 선택**:

- `--method direct`: train.py, train_irt.py 결과 평가용
- `--method drlagent`: train_finrl_standard.py 결과 평가용

### 권장 사용법

**시나리오별 스크립트 선택**:

1. **일반 RL 알고리즘 학습** → `train.py`

   - SAC, PPO 등 표준 알고리즘
   - IRT와 동일 조건 비교 가능

2. **IRT Policy 학습** → `train_irt.py`

   - 위기 적응형 포트폴리오
   - Alpha, eps 등 하이퍼파라미터 조정

3. **FinRL 표준 베이스라인** → `train_finrl_standard.py`

   - 논문 재현성 검증
   - DRLAgent 표준 파이프라인

4. **상세 평가** → `evaluate.py`
   - 모델 타입 자동 감지
   - 14개 시각화 (IRT) / 3개 (일반)

---

## 저장 위치

모든 학습/평가 결과는 `logs/` 아래에 타임스탬프로 저장된다.

### 디렉토리 구조

```
logs/
├── sac/                          # train.py 결과
│   └── 20251004_120000/
│       ├── checkpoints/
│       │   ├── sac_model_10000_steps.zip
│       │   ├── sac_model_20000_steps.zip
│       │   └── ...
│       ├── best_model/
│       │   └── best_model.zip
│       ├── sac_final.zip
│       ├── metadata.json         # 학습 시 사용된 ticker 및 기간 정보
│       ├── tensorboard/
│       │   └── events.out.tfevents.*
│       └── eval/
│           ├── evaluations.npz
│           └── evaluations.txt
│
├── ppo/                          # train.py 결과 (다른 모델)
│   └── 20251004_130000/
│       └── ...
│
├── finrl_sac/                    # train_finrl_standard.py 결과
│   └── 20251004_140000/
│       ├── sac_50k.zip           # 모델
│       ├── account_value_test.csv  # 평가 결과
│       ├── actions_test.csv      # 평가 결과
│       ├── logs/
│       │   ├── progress.csv
│       │   └── events.out.tfevents.*
│       └── tensorboard/
│
└── finrl_ppo/                    # train_finrl_standard.py 결과 (다른 모델)
    └── 20251004_150000/
        └── ...
```

### 파일 설명

| 파일                                    | 생성 스크립트           | 설명                                 |
| --------------------------------------- | ----------------------- | ------------------------------------ |
| `{model}_final.zip`                     | train.py                | 최종 모델 (학습 종료 시)             |
| `best_model/best_model.zip`             | train.py                | 최고 성능 모델 (EvalCallback)        |
| `checkpoints/{model}_model_*_steps.zip` | train.py                | 주기적 체크포인트 (10,000 steps마다) |
| `metadata.json`                         | train.py                | 학습 시 사용된 ticker 및 기간 정보 (평가 일관성) |
| `{model}_50k.zip`                       | train_finrl_standard.py | 모델 (50k timesteps)                 |
| `account_value_test.csv`                | train_finrl_standard.py | 평가 결과 (포트폴리오 가치)          |
| `actions_test.csv`                      | train_finrl_standard.py | 평가 결과 (행동 로그)                |
| `tensorboard/`                          | 공통                    | TensorBoard 로그                     |
| `logs/progress.csv`                     | train_finrl_standard.py | 학습 진행 로그                       |
| `eval/`                                 | train.py                | 평가 로그 (EvalCallback)             |

---

## 사용 시나리오

### 시나리오 1: FinRL 표준 베이스라인 검증

FinRL 논문과 동일한 조건으로 베이스라인을 학습 및 평가한다.

```bash
# 1. 학습 및 평가 (SAC, 50k timesteps)
python scripts/train_finrl_standard.py --model sac --mode both

# 결과 확인
ls logs/finrl_sac/20251004_*/
# sac_50k.zip
# account_value_test.csv
# actions_test.csv

# 2. 상세 평가 (DRLAgent 방식)
python scripts/evaluate.py \
  --model logs/finrl_sac/20251004_140000/sac_50k.zip \
  --method drlagent \
  --save-plot \
  --save-json

# 3. 시각화 및 메트릭 확인
cat logs/finrl_sac/20251004_140000/evaluation_results.json
open logs/finrl_sac/20251004_140000/evaluation_plots/portfolio_value.png
```

### 시나리오 2: IRT vs Baseline 동일 조건 비교

IRT Custom Policy와 정확히 동일한 조건으로 베이스라인을 학습한다.

```bash
# 1. Baseline 학습 (SAC, 200 episodes = 50k timesteps)
python scripts/train.py --model sac --mode both --episodes 200

# 2. IRT 학습 (동일 episodes)
python scripts/train_irt.py --episodes 200

# 3. 두 모델 평가 (동일 방식)
python scripts/evaluate.py \
  --model logs/sac/20251004_120000/sac_final.zip \
  --method direct \
  --save-json

python scripts/evaluate.py \
  --model logs/irt/20251004_120000/irt_final.zip \
  --method direct \
  --save-json

# 4. 결과 비교
diff logs/sac/20251004_120000/evaluation_results.json \
     logs/irt/20251004_120000/evaluation_results.json
```

### 시나리오 3: 다양한 알고리즘 비교

5가지 알고리즘을 모두 학습하고 비교한다.

```bash
# 1. 모든 알고리즘 학습 (병렬 가능)
for model in sac ppo a2c td3 ddpg; do
    python scripts/train.py --model $model --mode both --episodes 200 &
done
wait

# 2. 모든 모델 평가
for model_dir in logs/{sac,ppo,a2c,td3,ddpg}/20251004_*/; do
    python scripts/evaluate.py \
      --model ${model_dir}*_final.zip \
      --method direct \
      --save-json
done

# 3. 결과 취합 (Python 스크립트 별도 작성)
python scripts/compare_results.py logs/*/20251004_*/evaluation_results.json
```

### 시나리오 4: TensorBoard 모니터링

학습 중 TensorBoard로 실시간 모니터링한다.

```bash
# 1. 학습 시작 (백그라운드)
python scripts/train.py --model sac --mode train --episodes 200 &

# 2. TensorBoard 실행
tensorboard --logdir logs/sac/20251004_120000/tensorboard

# 3. 브라우저에서 확인
# http://localhost:6006
```

### 시나리오 5: 커스텀 기간 백테스트

COVID-19 위기 기간(2020)에 대한 성능을 별도 평가한다.

```bash
# 1. 일반 기간 학습 (2008-2019)
python scripts/train.py \
  --model sac \
  --mode train \
  --train-start 2008-01-01 \
  --train-end 2019-12-31 \
  --episodes 200

# 2. COVID-19 기간 평가 (2020)
python scripts/evaluate.py \
  --model logs/sac/20251004_120000/sac_final.zip \
  --test-start 2020-01-01 \
  --test-end 2020-12-31 \
  --save-plot \
  --save-json \
  --output-json results/covid_evaluation.json

# 3. 정상 기간 평가 (2021-2024)
python scripts/evaluate.py \
  --model logs/sac/20251004_120000/sac_final.zip \
  --test-start 2021-01-01 \
  --test-end 2024-12-31 \
  --save-plot \
  --save-json \
  --output-json results/normal_evaluation.json
```

---

## 팁 및 참고사항

### 학습 시간 예상

- **SAC (50k timesteps)**: 약 30-60분 (CPU), 10-20분 (GPU)
- **PPO (50k timesteps)**: 약 20-40분 (CPU), 8-15분 (GPU)
- 데이터 다운로드: 최초 1회 약 2-3분

### Checkpoint 활용

학습 중단 후 재개는 아직 지원하지 않는다. 향후 구현 예정.

현재는 `best_model/best_model.zip` 또는 `checkpoints/` 파일을 사용하여 평가만 가능하다.

### 메모리 사용량

- 학습: 약 2-4GB RAM
- 평가: 약 1-2GB RAM
- 대용량 데이터(2008-2024): 약 500MB

### 재현성

동일 결과를 얻으려면:

1. 동일한 `--train-start`, `--train-end` 사용
2. 동일한 `--episodes` 또는 `--timesteps` 사용
3. 동일한 `config.py` 하이퍼파라미터 사용
4. Seed 고정 (현재 미지원, 향후 추가 예정)

### 문제 해결

**Q: "ModuleNotFoundError: No module named 'finrl'"**

```bash
pip install -e .
```

**Q: "ValueError: Model type cannot be detected"**

```bash
python scripts/evaluate.py --model ... --model-type sac
```

**Q: "Data download failed"**

- 인터넷 연결 확인
- Yahoo Finance API 상태 확인
- `--train-end` 날짜가 미래가 아닌지 확인

---

## 관련 문서

- [IRT.md](IRT.md) - IRT 알고리즘 상세 설명
- [CHANGELOG.md](CHANGELOG.md) - 변경사항 이력
- [README.md](../README.md) - 프로젝트 개요
- [config.py](../finrl/config.py) - 설정 파일
- [FinRL 공식 문서](https://finrl.readthedocs.io/) - FinRL 라이브러리 참조
