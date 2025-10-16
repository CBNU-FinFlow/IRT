# IRT (Immune Replicator Transport) 기술 문서

> **위기 적응형 포트폴리오 관리를 위한 면역학 영감 강화학습 프레임워크**
>
> 마지막 업데이트: 2025-10-16

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [핵심 아이디어](#2-핵심-아이디어)
3. [시스템 아키텍처](#3-시스템-아키텍처)
4. [수학적 구성 요소](#4-수학적-구성-요소)
5. [위기 감지 시스템](#5-위기-감지-시스템)
6. [학습 및 평가](#6-학습-및-평가)
7. [사용 가이드](#7-사용-가이드)
8. [하이퍼파라미터 가이드](#8-하이퍼파라미터-가이드)
9. [성능 벤치마크](#9-성능-벤치마크)
10. [FAQ 및 문제 해결](#10-faq-및-문제-해결)
11. [참고 문헌](#11-참고-문헌)
12. [부록](#12-부록)

---

## 1. 프로젝트 개요

### 1.1 배경 및 동기

금융 시장은 **정상 구간**과 **위기 구간**에서 완전히 다른 특성을 보입니다:

- **정상 구간**: 낮은 변동성, 예측 가능한 패턴, 분산 투자 유리
- **위기 구간**: 높은 변동성, 급격한 변화, 상관관계 붕괴, 방어적 전략 필요

일반적인 강화학습 알고리즘은 **단일 정책**을 학습하므로, 두 구간 모두에 최적인 전략을 찾기 어렵습니다. IRT는 이 문제를 해결하기 위해 **적응 면역계(Adaptive Immune System)**에서 영감을 받은 메커니즘을 도입합니다.

### 1.2 IRT란?

**IRT (Immune Replicator Transport)**는 다음 세 가지 핵심 메커니즘을 결합한 위기 적응형 포트폴리오 관리 알고리즘입니다:

1. **T-Cell Crisis Detection**: 다중 신호 기반 실시간 위기 감지
2. **Optimal Transport (OT)**: 현재 시장 상태와 전문가 전략 간의 구조적 매칭
3. **Replicator Dynamics**: 과거 성공 전략에 대한 시간 메모리

### 1.3 주요 특징

- ✅ **위기 적응형**: 시장 상황에 따라 전략을 동적으로 조정
- ✅ **해석 가능**: 각 구성 요소의 기여도 시각화 가능
- ✅ **모듈화**: SB3 SAC와 완벽 통합, 다른 알고리즘 적용 가능
- ✅ **검증됨**: Dow Jones 30 종목, 2008-2024 데이터로 검증

### 1.4 시스템 요구사항

**환경**:

- Python 3.8+
- CUDA (선택사항, GPU 가속용)

**핵심 의존성**:

- `stable-baselines3>=2.0.0a5`: SAC 알고리즘
- `finrl==0.3.8`: 포트폴리오 환경
- `torch>=2.0.0`: 신경망 구현
- `yfinance`: 시장 데이터 다운로드

**데이터**:

- Yahoo Finance API를 통한 Dow Jones 30 종목
- 학습: 2009-2020 (12년)
- 평가: 2021-2024 (4년)

---

## 2. 핵심 아이디어

### 2.1 IRT 공식

IRT의 핵심은 **두 가지 경로의 동적 혼합**입니다:

```
w_t = (1-α_c)·Replicator(w_{t-1}, f_t, c_t) + α_c·Transport(E_t, K, C_t)
```

**구성 요소**:

- `w_t` [M]: M개 프로토타입(전문가 전략)의 혼합 가중치
- `α_c` [0,1]: 위기 적응형 혼합 비율
  - 평시(c≈0): α_c ≈ α_max (OT 지배, 균등 분산 선호)
  - 위기(c≈1): α_c ≈ α_min (Replicator 지배, fitness 기반 선택적 집중)
- `f_t` [M]: Critic Q-value 기반 프로토타입 적합도
- `E_t` [m×D]: 에피토프(상태를 m개 토큰으로 인코딩)
- `K` [M×D]: 학습 가능한 프로토타입 키

### 2.2 왜 효과적인가?

**1. 정상 구간 (c ≈ 0)**:

- α_c ≈ 0.45 (OT 비중 높음)
- OT가 모든 프로토타입을 고려하여 균등 분산
- 안정적 수익 추구

**2. 위기 구간 (c ≈ 1)**:

- α_c ≈ 0.08 (Replicator 비중 높음)
- Replicator가 Q-value 높은 프로토타입에 집중
- 빠른 적응 (η ↑), 검증된 전략 선택

**3. 다중 신호 위기 감지**:

```
crisis = w_r·T-Cell(시장통계+기술지표)
       + w_s·DSR(성능변화)
       + w_c·CVaR(꼬리위험)
```

### 2.3 면역학적 비유

| 생물학               | IRT               | 역할                          |
| -------------------- | ----------------- | ----------------------------- |
| **T-Cell**           | Crisis Detector   | 위협(위기) 감지 및 신호 방출  |
| **B-Cell**           | IRT Actor         | 항체(포트폴리오) 생성         |
| **Antigen**          | Market State      | 외부 환경 정보                |
| **Epitope**          | State Tokens      | 항원의 인식 가능 부위         |
| **Antibody**         | Expert Strategies | 특정 상황에 특화된 전략       |
| **Affinity**         | Cost Matrix       | 전략-상황 간 적합도           |
| **Clonal Selection** | IRT Mixing        | 상황에 맞는 전략 선택 및 혼합 |

> **주의**: 이는 직관을 돕기 위한 비유입니다. IRT는 강화학습 프레임워크이며, "방어"는 학습된 Q-function의 결과입니다.

---

## 3. 시스템 아키텍처

### 3.1 전체 구조 다이어그램

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          SAC with IRT Actor                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│                           ┌───────────────────┐                            │
│                      ┌───►│   Environment     │────┐                       │
│                      │    │      (MDP)        │    │                       │
│                      │    └───────────────────┘    │                       │
│                  a_t │                             │ s_t+1                 │
│                      │                             │                       │
│            ┌─────────┴──────────┐                  │                       │
│            │    Actor π_θ       │                  │                       │
│            │      (IRT)         │                  │                       │
│            │                    │                  │                       │
│            │  ┌──────────────┐  │                  │                       │
│            │  │  T-Cell Det  │  │                  │                       │
│            │  └──────┬───────┘  │                  │                       │
│            │         ▼          │                  │                       │
│            │   Epitope Encoder  │                  │                       │
│            │         ▼          │         (s,a,r,s',d)                     │
│            │   IRT Operator     │                  │                       │
│            │    (OT+Rep)        │                  ▼                       │
│            │         ▼          │        ┌──────────────────┐              │
│            │  Dirichlet Mixture │        │  Replay Buffer   │              │
│            └────────────────────┘        │   D={(s,a,r,..)} │              │
│                      ▲                   └─────────┬────────┘              │
│                      │                             │                       │
│                      │                     sample mini-batch               │
│                      │                             │                       │
│                      │                             ▼                       │
│                      │                   ┌──────────────────┐              │
│                      │                   │ Twin Critics     │              │
│                      │                   │   Q_ϕ1, Q_ϕ2     │              │
│                      │                   └─────────┬────────┘              │
│                      │                             │                       │
│                      │                             │ Q(s, π_j(s))          │
│                      │                             │   (fitness)           │
│                      └─────────────────────────────┘                       │
│                                                                            │
│  Training Update:                                                          │
│    Critic:  min 𝔼[(Q_ϕ(s,a) - (r + γQ_target(s',a')))²]                    │
│    Actor:   max 𝔼[Q_ϕ(s, π_θ(s)) - α·log π_θ(a|s)]                         │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 IRT Actor 상세 구조

```
                    Input State s_t ∈ ℝ^d  (d=181)
                           │
      ┌────────────────────┼────────────────────┐
      │                    │                    │
      ▼                    ▼                    ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Market     │    │   State      │    │  Q-value        │
│  Features   │    │   Features   │    │  Estimation     │
│  [12]       │    │   [181]      │    │  (Twin Critics) │
└──────┬──────┘    └──────┬───────┘    └────────┬────────┘
       │                  │                     │
       ▼                  ▼                     ▼
┌──────────────┐   ┌─────────────┐     Fitness [B,M]
│  T-Cell      │   │  Epitope    │             │
│  Detector    │   │  Encoder    │             │
│              │   │             │             │
│ crisis_level │   │  E [B,m,D]  │             │
│ danger [B,D] │   └─────────────┘             │
└──────┬───────┘                               │
       │                                        │
       │    Prototype Keys K [M,D]              │
       │            │                           │
       └────────────┼───────────────────────────┘
                    │
                    ▼
       ┌────────────────────────────┐
       │    IRT Operator            │
       │    ───────────────          │
       │                            │
       │  ┌──────────────────────┐  │
       │  │ Optimal Transport    │  │
       │  │ ─────────────────    │  │
       │  │ Cost Matrix C        │  │
       │  │ Sinkhorn(C, ε)      │  │
       │  │ → w_ot [B,M]        │  │
       │  └──────────────────────┘  │
       │                            │
       │  ┌──────────────────────┐  │
       │  │ Replicator Dynamics  │  │
       │  │ ───────────────────  │  │
       │  │ η(c) = η_0 + η_1·c  │  │
       │  │ A = fitness - mean  │  │
       │  │ → w_rep [B,M]       │  │
       │  └──────────────────────┘  │
       │                            │
       │  ┌──────────────────────┐  │
       │  │ Dynamic Mixing       │  │
       │  │ ─────────────────    │  │
       │  │ α_c = f(crisis, Δs) │  │
       │  │ w = (1-α)·w_rep +   │  │
       │  │     α·w_ot          │  │
       │  └──────────────────────┘  │
       └────────────┬───────────────┘
                    │
                    ▼ w [B,M]
       ┌────────────────────────────┐
       │  Dirichlet Mixture         │
       │  ─────────────────          │
       │                            │
       │  For j=1...M:              │
       │    decoder_j(K_j) → α_j    │
       │                            │
       │  mixed_α = Σ w_j·α_j       │
       │                            │
       │  action ~ Dirichlet(α)     │
       │  or softmax(α/τ)           │
       └────────────┬───────────────┘
                    │
                    ▼
            Portfolio Weights [B,n]
```

### 3.3 데이터 흐름 요약

| 단계                | 입력                                  | 출력                 | 차원            |
| ------------------- | ------------------------------------- | -------------------- | --------------- |
| **Environment**     | action                                | state                | [n] → [d]       |
| **Market Features** | state                                 | market_features      | [d] → [12]      |
| **T-Cell**          | market_features                       | crisis_level, danger | [12] → [1], [D] |
| **Epitope Encoder** | state                                 | E                    | [d] → [m,D]     |
| **Proto Keys**      | -                                     | K                    | [M,D]           |
| **Q-Networks**      | state, action                         | fitness              | [d],[n] → [M]   |
| **IRT Operator**    | E, K, danger, w_prev, fitness, crisis | w                    | → [M]           |
| **Decoders**        | K                                     | concentrations       | [M,D] → [M,n]   |
| **Mixture**         | w, concentrations                     | action               | [M],[M,n] → [n] |

**차원 범례**:

- `d=181`: 상태 차원
- `n=30`: 행동 차원 (자산 수)
- `m=6`: 에피토프 토큰 수
- `M=8`: 프로토타입 수
- `D=128`: 임베딩 차원

### 3.4 주요 모듈 개요

| 모듈                | 파일                  | 역할             | 핵심 메서드/속성                                       |
| ------------------- | --------------------- | ---------------- | ------------------------------------------------------ |
| **IRTPolicy**       | `irt_policy.py`       | SB3 SAC 통합     | `make_actor()`, `get_irt_info()`                       |
| **IRTActorWrapper** | `irt_policy.py`       | Actor 인터페이스 | `forward()`, `action_log_prob()`, `_compute_fitness()` |
| **BCellIRTActor**   | `bcell_actor.py`      | IRT 핵심 구현    | `forward()`, 6-step process                            |
| **IRT Operator**    | `irt_operator.py`     | OT + Replicator  | `forward()`, `_cost_matrix()`                          |
| **TCellMinimal**    | `t_cell.py`           | 위기 감지        | `forward()`, EMA normalization                         |
| **StockTradingEnv** | `env_stocktrading.py` | 환경             | `step()`, `reset()`                                    |

---

## 4. 수학적 구성 요소

### 4.1 IRT 전체 수식

\[
\begin{aligned}
\text{[위기 감지]} \quad &c*t = \text{MultiSignal}(x^m_t, \Delta s_t, \text{CVaR}\_t) \\
\text{[에피토프]} \quad &E_t = \phi_s(s_t) \in \mathbb{R}^{m \times D} \\
\text{[비용 행렬]} \quad &C*{ij} = d*M(E_i, K_j) - \gamma \langle E_i, d_t \rangle + \lambda*{tol} \mathcal{T}_i + \rho \cdot \text{conf}\_j \\
\text{[OT]} \quad &P^*\_t = \arg\min_P \langle P, C \rangle + \varepsilon \text{KL}(P \| u \otimes v^\top), \quad w^{ot}\_t = P^*\_t \mathbf{1}\_m \\
\text{[Replicator]} \quad &\eta(c_t) = \eta_0 + \eta_1 c_t, \quad \tilde{w}\_t \propto w_{t-1} \cdot \exp\{\eta(c*t)(f_t - \bar{f}\_t) - r_t\} \\
\text{[동적 혼합]} \quad &\alpha_c = \alpha*{\max} + (\alpha*{\min} - \alpha*{\max}) \frac{1 - \cos(\pi c*t)}{2} \\
\text{[융합]} \quad &w_t = (1-\alpha_c) w^{rep}\_t + \alpha_c w^{ot}\_t \\
\text{[Dirichlet]} \quad &\alpha_t = \sum*{j=1}^M w\_{t,j} \cdot \alpha_j(K_j), \quad a_t \sim \text{Dirichlet}(\alpha_t) \text{ or } \text{softmax}(\alpha_t/\tau)
\end{aligned}
\]

### 4.2 Optimal Transport (Sinkhorn)

**목적**: 에피토프 $E \in \mathbb{R}^{m \times D}$와 프로토타입 $K \in \mathbb{R}^{M \times D}$ 간의 최적 수송

**최적화 문제**:
\[
P^\* = \arg\min*{P \in \mathbb{R}^{m \times M}*+} \langle P, C \rangle + \varepsilon H(P)
\]
\[
\text{s.t.} \quad P \mathbf{1}\_M = u, \quad P^\top \mathbf{1}\_m = v
\]

**Sinkhorn 반복** ($K = \exp(-C/\varepsilon)$):
\[
a \leftarrow u \oslash (K b), \quad b \leftarrow v \oslash (K^\top a), \quad P^\* = \text{diag}(a) K \text{diag}(b)
\]

**OT 마진**:
\[
w^{ot}_j = \sum_{i=1}^m P^\*\_{ij}
\]

**파라미터**:

- $\varepsilon = 0.03$: 엔트로피 정규화 (exploration)
- $u = v = \frac{1}{m}\mathbf{1}$: 균등 마진

### 4.3 Replicator Dynamics

**기원**: 진화 게임 이론 (Hofbauer & Sigmund, 1998)

**IRT 이산 버전**:
\[
\log \tilde{w}_{t,j} = \log w_{t-1,j} + \eta(c*t) \cdot (f*{t,j} - \bar{f}_t) - r_{t,j}
\]
\[
w^{rep}_{t,j} = \frac{\exp(\log \tilde{w}_{t,j} / \tau)}{\sum*k \exp(\log \tilde{w}*{t,k} / \tau)}
\]

**위기 가열**:
\[
\eta(c_t) = \eta_0 + \eta_1 c_t
\]

- 평시 ($c_t \approx 0$): $\eta \approx 0.05$
- 위기 ($c_t \approx 1$): $\eta \approx 0.17$

**파라미터**:

- $\eta_0 = 0.05$, $\eta_1 = 0.12$
- $\tau = 1.4$: Temperature (분포 평탄화)

### 4.4 면역학적 비용 함수

\[
C*{ij} = \underbrace{d_M(E_i, K_j)}*{\text{거리}} - \underbrace{\gamma \langle E*i, d_t \rangle}*{\text{공자극}} + \underbrace{\lambda*{tol} \mathcal{T}\_i}*{\text{내성}} + \underbrace{\rho \cdot \text{conf}_j}_{\text{체크포인트}}
\]

**구성 요소**:

1. **마할라노비스 거리**: $d_M(x, y) = \sqrt{(x-y)^\top M (x-y)}$, $M = L^\top L$ (학습 가능)
2. **공자극**: $\langle E_i, d_t \rangle$ (위기 신호와 정렬 시 비용 ↓)
3. **내성**: $\mathcal{T}_i = \text{ReLU}(\kappa \cdot \max_k \langle E_i, \sigma_k \rangle - \varepsilon_{tol})$ (실패 패턴 억제)
4. **체크포인트**: 과신 프로토타입 억제

**파라미터**:

- $\gamma = 0.85$, $\lambda_{tol} = 2.0$, $\rho = 0.3$
- $\kappa = 1.0$, $\varepsilon_{tol} = 0.1$

### 4.5 동적 혼합 계수

**위기 기반**:
\[
\alpha*c(c_t) = \alpha*{\max} + (\alpha*{\min} - \alpha*{\max}) \frac{1 - \cos(\pi c_t)}{2}
\]

**Sharpe 피드백**:
\[
\alpha'\_c = \alpha_c \cdot (1 + 0.25 \cdot \tanh(\Delta s_t))
\]

**파라미터**:

- $\alpha_{\min} = 0.08$, $\alpha_{\max} = 0.45$

---

## 5. 위기 감지 시스템

### 5.1 다중 신호 통합

IRT는 세 가지 독립 신호를 결합:

```
crisis_raw = w_r · sigmoid(k_b · z_base)      # T-Cell 시장 신호
           + w_s · sigmoid(-k_s · z_sharpe)    # DSR gradient
           + w_c · sigmoid(k_c · z_cvar)       # CVaR
```

**기본 가중치**:

- `w_r = 0.55`: T-Cell (주요 센서)
- `w_s = -0.25`: DSR (음수: 하락=위기)
- `w_c = 0.20`: CVaR (꼬리 위험)

### 5.2 T-Cell 구조

**입력** [12차원]:

- 시장 통계 (4개): balance, price_mean, price_std, cash_ratio
- 기술 지표 (8개): macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, sma_30, sma_60

**출력**:

- `crisis_types` [K=4]: 위기 타입별 점수
- `danger_embed` [D]: 공자극 임베딩
- `crisis_base`: 기본 위기 확률

**온라인 정규화**:
\[
z\_{std} = \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}}
\]

- EMA momentum = 0.92

### 5.3 바이어스/온도 보정

목표 위기 점유율 (`p_star=0.35`)로 보정:

```python
crisis_affine = (crisis_raw - bias) / temperature
crisis_pre = sigmoid(crisis_affine)

if training:
    p_hat = crisis_pre.mean()
    bias += η_b · (p_hat - p_star)
    temperature *= (1 + η_T · (p_hat - p_star))
```

**파라미터**:

- `η_b`: 0.02 → 0.002 (cosine decay, 30k steps)
- `η_T = 0.01`
- `temperature_range = [0.9, 1.2]`

### 5.4 T-Cell Guard

목표 레짐 점유율로 유도:

```python
crisis_level = crisis_pre + guard_rate · (target - crisis_pre)
```

**파라미터**:

- `crisis_target = 0.5`
- `guard_rate_init = 0.07` → `guard_rate_final = 0.02` (7500 steps)

### 5.5 히스테리시스

레짐 전환 안정화:

```python
if prev_regime == CRISIS:
    regime = (crisis < 0.45) ? NORMAL : CRISIS
else:
    regime = (crisis > 0.55) ? CRISIS : NORMAL
```

---

## 6. 학습 및 평가

### 6.1 학습 파이프라인

**스크립트**: `scripts/train_irt.py`

**기본 명령**:

```bash
python scripts/train_irt.py --mode train --episodes 200
```

**주요 단계**:

1. 데이터 다운로드 (Yahoo Finance, Dow Jones 30)
2. 기술 지표 계산 (MACD, Bollinger, RSI, CCI, DX, SMA)
3. 환경 생성 (`StockTradingEnv`, `reward_type=adaptive_risk`)
4. SAC + IRTPolicy 학습
5. 체크포인트 저장 (`logs/irt/<timestamp>/`)

**학습 설정**:

- Buffer size: 100,000
- Batch size: 256
- Learning rate: 1e-4
- Episodes: 200 (약 50,000 timesteps)

### 6.2 평가

**스크립트**: `scripts/evaluate.py`

**기본 명령**:

```bash
python scripts/evaluate.py \
  --model logs/irt/20251005_123456/irt_final.zip \
  --method direct \
  --save-plot \
  --save-json
```

**평가 모드**:

- `direct`: SB3 모델 직접 로드 (IRT 정보 포함)
- `drlagent`: FinRL DRLAgent API (SAC baseline 비교용)

**생성 결과물**:

- `evaluation_results.json`: 10개 메트릭
- `evaluation_plots/`: 14개 시각화 (IRT 특화 포함)
- `xai/`: XAI 분석 (Integrated Gradients)

### 6.3 시각화

**IRT 특화 플롯** (11개):

1. `irt_decomposition.png`: OT vs Replicator 기여도
2. `portfolio_weights.png`: 시간별 포트폴리오 가중치
3. `crisis_levels.png`: 위기 레벨 추이
4. `prototype_weights.png`: 프로토타입 활성화 패턴
5. `stock_analysis.png`: 종목별 비중 분석
6. `performance_timeline.png`: 성능 시계열
7. `benchmark_comparison.png`: 벤치마크 비교
8. `risk_dashboard.png`: 위험 지표 대시보드
9. `tcell_analysis.png`: T-Cell 위기 타입 분석
10. `attribution_analysis.png`: 성능 기여도 분석
11. `cost_matrix.png`: OT 비용 행렬 히트맵

**일반 플롯** (3개):

1. `portfolio_value.png`: 포트폴리오 가치 추이
2. `returns_distribution.png`: 수익률 분포
3. `drawdown.png`: Drawdown 추이

### 6.4 모델 비교

**스크립트**: `scripts/compare_models.py`

```bash
python scripts/compare_models.py \
  --model1 logs/sac \
  --model2 logs/irt \
  --output comparison_results
```

**생성 결과**:

- `comparison_summary.json`: 비교 메트릭
- `plots/`: 8개 비교 플롯
  - portfolio_value_comparison
  - drawdown_comparison
  - performance_metrics
  - risk_metrics
  - rolling_sharpe
  - crisis_response (IRT only)

---

## 7. 사용 가이드

### 7.1 빠른 시작

**1. 설치**:

```bash
pip install -e .
```

**2. 학습**:

```bash
python scripts/train_irt.py --mode train --episodes 200
```

**3. 평가**:

```bash
python scripts/train_irt.py --mode eval \
  --model logs/irt/<timestamp>/irt_final.zip
```

**4. 통합 (학습 + 평가)**:

```bash
python scripts/train_irt.py --mode both --episodes 200
```

### 7.2 Python API 사용

```python
from stable_baselines3 import SAC
from finrl.agents.irt import IRTPolicy
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# 1. 환경 생성
env = StockTradingEnv(
    df=train_data,
    reward_type='adaptive_risk',
    use_weighted_action=True,
    ...
)

# 2. IRT Policy 설정
policy_kwargs = {
    "emb_dim": 128,
    "m_tokens": 6,
    "M_proto": 8,
    "alpha_min": 0.08,
    "alpha_max": 0.45,
    "eta_0": 0.05,
    "eta_1": 0.12,
    "eps": 0.03,
    ...
}

# 3. SAC + IRT 모델 생성
model = SAC(
    policy=IRTPolicy,
    env=env,
    policy_kwargs=policy_kwargs,
    learning_rate=1e-4,
    buffer_size=100000,
    batch_size=256,
    verbose=1
)

# 4. 학습
model.learn(total_timesteps=50000)

# 5. 저장
model.save("irt_model.zip")

# 6. 평가
obs = env.reset()
for _ in range(len(test_data)):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        break

# 7. IRT 정보 추출
irt_info = model.policy.get_irt_info()
print(f"위기 레벨: {irt_info['crisis_level']}")
print(f"프로토타입 가중치: {irt_info['w']}")
print(f"동적 α: {irt_info['alpha_c']}")
```

### 7.3 CLI 인자

**학습 관련**:

- `--episodes`: 에피소드 수 (기본: 200)
- `--seed`: 랜덤 시드 (기본: 42)
- `--reward-type`: 보상 함수 (`adaptive_risk`, `dsr_cvar`, `basic`)
- `--env-diversify`: 환경 다각화 (`none`, `rolling`, `random`)
- `--domain-rand-tx`: 거래 비용 랜덤화 (±%)
- `--domain-rand-slippage`: 슬리피지 랜덤화 (±%)

**IRT 파라미터**:

- `--alpha-min`, `--alpha-max`: α 범위
- `--eta-0`, `--eta-1`: Replicator 학습률
- `--eps`: Sinkhorn 엔트로피
- `--gamma`: 공자극 가중치
- `--w-r`, `--w-s`, `--w-c`: 위기 신호 가중치

**평가 관련**:

- `--method`: 평가 모드 (`direct`, `drlagent`)
- `--xai-level`: XAI 분석 수준 (`none`, `basic`, `full`)
- `--save-plot`, `--save-json`: 결과 저장

### 7.4 디버깅

**1. TensorBoard 로그 확인**:

```bash
tensorboard --logdir logs/irt/<timestamp>/tensorboard
```

**주요 메트릭**:

- `train/alpha_c_mean`: 동적 α 평균
- `train/crisis_level_mean`: 위기 레벨 평균
- `train/w_rep_entropy`: Replicator 엔트로피
- `train/w_ot_entropy`: OT 엔트로피

**2. IRT 정보 로깅**:

```python
irt_info = model.policy.get_irt_info()

# 위기 감지
print(f"Crisis Level: {irt_info['crisis_level'].mean():.3f}")
print(f"Crisis Regime: {irt_info['crisis_regime'].mean():.3f}")

# IRT 분해
print(f"w_rep: {irt_info['w_rep'].mean(0)}")
print(f"w_ot: {irt_info['w_ot'].mean(0)}")
print(f"w (mixed): {irt_info['w'].mean(0)}")

# 동적 α
print(f"alpha_c: {irt_info['alpha_c'].mean():.3f}")

# Fitness
print(f"Fitness: {irt_info['fitness'].mean(0)}")
```

**3. 단위 테스트**:

```bash
pytest tests/test_irt_policy.py -v
```

---

## 8. 하이퍼파라미터 가이드

### 8.1 핵심 파라미터

| 파라미터          | 기본값 | 범위      | 설명               | 영향             |
| ----------------- | ------ | --------- | ------------------ | ---------------- |
| **구조**          |
| `emb_dim`         | 128    | 64-256    | 임베딩 차원        | 표현력 vs 연산량 |
| `m_tokens`        | 6      | 4-8       | 에피토프 토큰 수   | 상태 다중 표현   |
| `M_proto`         | 8      | 6-12      | 프로토타입 수      | 전문가 다양성    |
| **동적 혼합**     |
| `alpha_min`       | 0.08   | 0.05-0.15 | 위기 시 최소 α     | Replicator 비중  |
| `alpha_max`       | 0.45   | 0.35-0.55 | 평시 최대 α        | OT 비중          |
| `ema_beta`        | 0.70   | 0.5-0.85  | w_prev EMA 계수    | 시간 메모리      |
| **위기 감지**     |
| `w_r`             | 0.55   | 0.4-0.7   | T-Cell 가중치      | 시장 신호 민감도 |
| `w_s`             | -0.25  | -0.4~-0.1 | DSR 가중치         | 성능 하락 민감도 |
| `w_c`             | 0.20   | 0.1-0.3   | CVaR 가중치        | 꼬리 위험 민감도 |
| **Replicator**    |
| `eta_0`           | 0.05   | 0.03-0.08 | 기본 학습률        | 평시 적응 속도   |
| `eta_1`           | 0.12   | 0.05-0.20 | 위기 증가량        | 위기 적응 속도   |
| `replicator_temp` | 1.4    | 0.9-2.0   | Softmax 온도       | 분포 평탄화      |
| **OT**            |
| `eps`             | 0.03   | 0.01-0.1  | 엔트로피 정규화    | Exploration      |
| `gamma`           | 0.85   | 0.5-1.0   | 공자극 가중치      | 위기 신호 민감도 |
| `lambda_tol`      | 2.0    | 1.0-3.0   | 내성 가중치        | 실패 회피        |
| **Dirichlet**     |
| `dirichlet_min`   | 0.8    | 0.1-1.0   | Concentration 최소 | Sparsity         |
| `dirichlet_max`   | 20.0   | 5.0-50.0  | Concentration 최대 | Peakedness       |
| `action_temp`     | 0.8    | 0.3-1.0   | 결정적 행동 온도   | 평가 시 집중도   |

### 8.2 Ablation Study 예시

**1. α 범위 조정**:

```bash
# 위기 시 Replicator 비중 증가
python scripts/train_irt.py --alpha-min 0.05 --alpha-max 0.55

# 위기 시 Replicator 비중 감소
python scripts/train_irt.py --alpha-min 0.12 --alpha-max 0.35
```

**2. 위기 신호 가중치 조정**:

```bash
# T-Cell 신호 강화
python scripts/train_irt.py --w-r 0.70 --w-s -0.15 --w-c 0.15

# DSR 신호 강화 (성능 하락 민감)
python scripts/train_irt.py --w-r 0.45 --w-s -0.35 --w-c 0.20
```

**3. Replicator 적응 속도**:

```bash
# 빠른 적응 (불안정 위험)
python scripts/train_irt.py --eta-1 0.20

# 느린 적응 (안정적)
python scripts/train_irt.py --eta-1 0.08
```

**4. OT Exploration**:

```bash
# 높은 엔트로피 (탐색 증가)
python scripts/train_irt.py --eps 0.05

# 낮은 엔트로피 (집중 증가)
python scripts/train_irt.py --eps 0.01
```

### 8.3 파라미터 Phase 진화

IRT는 여러 실험을 통해 파라미터를 보정했습니다:

| Phase         | 주요 변경                           | 목적                     |
| ------------- | ----------------------------------- | ------------------------ |
| **Phase 1**   | 다중 신호 (w_r, w_s, w_c)           | T-Cell + DSR + CVaR 통합 |
| **Phase 2**   | action_temp=0.8                     | 결정적 행동 민감도       |
| **Phase 3**   | ema_beta=0.70                       | 전달 감쇠 완화           |
| **Phase 3.5** | dirichlet_min=0.8, alpha_min=0.05   | 균등 흡인 완화           |
| **Phase B**   | 바이어스/온도 보정                  | 위기 점유율 안정화       |
| **Phase E**   | eta_1=0.12, gamma=0.85              | Replicator 민감도 완화   |
| **Phase F**   | alpha_max=0.45, replicator_temp=1.4 | OT 증가, 분포 평탄화     |

---

## 9. 성능 벤치마크

### 9.1 목표 메트릭

| 메트릭                | SAC Baseline | IRT 목표 | 개선율  |
| --------------------- | ------------ | -------- | ------- |
| **Sharpe Ratio**      | 1.0-1.2      | 1.3-1.5  | +15-20% |
| **전체 Max Drawdown** | -30~-35%     | -18~-23% | -25-35% |
| **위기 구간 MDD**     | -40~-45%     | -22~-27% | -35-45% |
| **Calmar Ratio**      | 2.5-3.0      | 3.5-4.0  | +30-40% |

### 9.2 위기 구간 정의

- **2020 COVID-19**: 2020-03 ~ 2020-06
- **2022 Fed 금리 인상**: 2022-01 ~ 2022-12

### 9.3 실험 결과 (예시)

**전체 구간** (2021-2024):

- Sharpe: 1.35
- Calmar: 3.78
- Max Drawdown: -21.3%
- Annual Return: 18.2%

**위기 구간** (2020-03 ~ 2020-06):

- Max Drawdown: -24.5% (SAC: -42.1%)
- Recovery Time: 45일 (SAC: 78일)

**정상 구간** (2019-01 ~ 2019-12):

- Sharpe: 1.42 (SAC: 1.38)
- Annual Return: 16.5% (SAC: 15.8%)

### 9.4 해석 가능성

**IRT 분해 예시**:

- 평시 위기 레벨: 0.28 ± 0.12
- 위기 시 위기 레벨: 0.73 ± 0.15
- 평시 α_c: 0.38 ± 0.08 (OT 38%)
- 위기 시 α_c: 0.11 ± 0.05 (OT 11%, Replicator 89%)

**프로토타입 활성화**:

- 정상 구간: 프로토타입 2, 5, 7 활성 (분산 투자)
- 위기 구간: 프로토타입 1, 4 활성 (방어적 전략)

---

## 10. FAQ 및 문제 해결

### 10.1 일반 질문

**Q1: IRT는 어떤 알고리즘과 호환되나요?**

A: 기본적으로 SAC와 최적화되어 있지만, Q-network 기반 알고리즘(TD3, DDPG)과도 호환 가능합니다. PPO/A2C는 V(s) 기반이라 수정 필요.

**Q2: α ↓는 보수적 전략을 의미하나요?**

A: **아닙니다**. α ↓는 Replicator 비중 증가를 의미하며, 이는 Q-value 높은 프로토타입에 "집중"하는 것입니다. 이것이 방어적인지는 학습된 Q-function에 달려있습니다.

**Q3: 다른 자산 클래스에 적용 가능한가요?**

A: 네, 암호화폐, 채권, 상품 등에 적용 가능합니다. 환경의 `df` 데이터만 변경하면 됩니다.

**Q4: 학습 시간은 얼마나 걸리나요?**

A: GPU (RTX 3090) 기준 약 2-3시간 (200 episodes). CPU는 8-12시간.

**Q5: IRT 정보를 실시간으로 로깅하려면?**

A: Custom callback 사용:

```python
from stable_baselines3.common.callbacks import BaseCallback

class IRTLoggingCallback(BaseCallback):
    def _on_step(self):
        if self.n_calls % 100 == 0:
            irt_info = self.model.policy.get_irt_info()
            self.logger.record("irt/crisis_level", irt_info['crisis_level'].mean())
            self.logger.record("irt/alpha_c", irt_info['alpha_c'].mean())
        return True

model.learn(total_timesteps=50000, callback=IRTLoggingCallback())
```

### 10.2 문제 해결

**Q1: NaN 발생 (loss/Q-value)**

**원인**: 수치 불안정 (Sinkhorn, Replicator)

**해결**:

1. Sinkhorn tolerance 증가: `tol=1e-4` → `1e-3`
2. Replicator clamp 강화: `w_prev + 1e-8` → `w_prev + 1e-6`
3. Learning rate 감소: `1e-4` → `5e-5`

**Q2: α가 좁은 범위에 고착**

**원인**: Reward 함수와 T-Cell 방향 충돌

**해결**: Adaptive-risk reward의 `crisis_gain` 부호 반전

```python
# reward_functions.py
crisis_gain = -0.15  # 0.25 → -0.15
```

**Q3: 위기 감지가 너무 민감/둔감**

**해결**:

- 민감: `k_b`, `k_s`, `k_c` 감소 (4.0 → 3.0)
- 둔감: 위 값 증가 (4.0 → 5.0)
- 또는 가중치 조정 (`w_r`, `w_s`, `w_c`)

**Q4: Replicator가 균등 분포에 고착**

**원인**: Temperature 너무 높음

**해결**: `replicator_temp` 감소 (1.4 → 1.0)

**Q5: 학습 초기 불안정**

**해결**:

1. Replay buffer warmup 증가: `learning_starts=10000`
2. Crisis guard rate 증가: `crisis_guard_rate_init=0.15`
3. EMA beta 증가: `ema_beta=0.80`

### 10.3 Reward 함수 충돌 문제

**⚠️ 현재 문제**:

```
T-Cell:  위기 → α↓ → Replicator 활성 → 집중
Reward:  위기 → κ↑ → Sharpe bonus 증가 → 고수익 추구

→ 충돌! α가 좁은 범위에 고착
```

**권장 해결책**:

`finrl/meta/env_stock_trading/reward_functions.py` 수정:

```python
# 기존
self.adaptive_crisis_gain_sharpe = 0.25  # ❌

# 수정
self.adaptive_crisis_gain_sharpe = -0.15  # ✅
```

**효과**:

- κ(c) = 0.20 - 0.15\*c
- 평시: κ=0.20 (Sharpe 추구)
- 위기: κ=0.05 (생존 우선)
- T-Cell과 Reward 일치 → α 넓은 범위 활용

---

## 11. 참고 문헌

### 11.1 이론적 기초

1. **Cuturi, M. (2013)**. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport". _NIPS 2013_.

   - IRT의 Optimal Transport 메커니즘 기반

2. **Hofbauer, J., & Sigmund, K. (1998)**. "Evolutionary Games and Population Dynamics". _Cambridge University Press_.

   - IRT의 Replicator Dynamics 이론적 근거

3. **Amari, S. (2016)**. "Information Geometry and Its Applications". _Applied Mathematical Sciences_.
   - Dirichlet 분포 및 정보기하학

### 11.2 강화학습 프레임워크

4. **Liu, X. Y., et al. (2024)**. "FinRL: Financial Reinforcement Learning Framework". _NeurIPS Workshop_.

   - IRT 통합 포트폴리오 환경

5. **Raffin, A., et al. (2021)**. "Stable-Baselines3: Reliable Reinforcement Learning Implementations". _JMLR_.
   - SAC 알고리즘 구현 기반

### 11.3 금융 응용

6. **Markowitz, H. (1952)**. "Portfolio Selection". _Journal of Finance_.

   - 현대 포트폴리오 이론

7. **Rockafellar, R. T., & Uryasev, S. (2000)**. "Optimization of conditional value-at-risk". _Journal of Risk_.
   - CVaR 개념

### 11.4 면역학

8. **Janeway, C. A., et al. (2001)**. "Immunobiology". _Garland Science_.
   - 적응 면역계 메커니즘

### 11.5 추가 자료

- **프로젝트 README**: [../README.md](../README.md)
- **변경사항 이력**: [CHANGELOG.md](CHANGELOG.md)
- **스크립트 가이드**: [SCRIPTS.md](SCRIPTS.md)
- **FinRL 공식 문서**: https://finrl.readthedocs.io/
- **Stable Baselines3 문서**: https://stable-baselines3.readthedocs.io/

---

## 12. 부록

### 12.1 코드 매핑

| 기능           | 파일                                               | 주요 클래스/함수               |
| -------------- | -------------------------------------------------- | ------------------------------ |
| **IRT 정책**   | `finrl/agents/irt/irt_policy.py`                   | `IRTPolicy`, `IRTActorWrapper` |
| **IRT Actor**  | `finrl/agents/irt/bcell_actor.py`                  | `BCellIRTActor`                |
| **IRT 연산자** | `finrl/agents/irt/irt_operator.py`                 | `IRT`, `_cost_matrix()`        |
| **위기 감지**  | `finrl/agents/irt/t_cell.py`                       | `TCellMinimal`                 |
| **환경**       | `finrl/meta/env_stock_trading/env_stocktrading.py` | `StockTradingEnv`              |
| **보상 함수**  | `finrl/meta/env_stock_trading/reward_functions.py` | `RiskSensitiveReward`          |
| **학습**       | `scripts/train_irt.py`                             | `main()`                       |
| **평가**       | `scripts/evaluate.py`                              | `evaluate_model()`             |
| **비교**       | `scripts/compare_models.py`                        | `compare_models()`             |
| **테스트**     | `tests/test_irt_policy.py`                         | `test_*`                       |

### 12.2 용어 대조표

| 논문 용어 (Formal)   | 코드베이스 용어 (Metaphorical) | 위치              |
| -------------------- | ------------------------------ | ----------------- |
| Crisis Detector      | T-Cell Network                 | `t_cell.py`       |
| State Encoder        | Epitope Encoder                | `bcell_actor.py`  |
| Expert Strategies    | Prototype Keys                 | `bcell_actor.py`  |
| Expert Policy Heads  | Decoders                       | `bcell_actor.py`  |
| Q-values             | Fitness                        | `irt_policy.py`   |
| State Representation | Epitope Tokens                 | `bcell_actor.py`  |
| Crisis Embedding     | Danger Embedding               | `bcell_actor.py`  |
| Expert Weights       | Prototype Weights              | `irt_operator.py` |
| Distance Matrix      | Immunological Cost Matrix      | `irt_operator.py` |

### 12.3 환경 변수

**선택적 환경 변수**:

```bash
export CUDA_VISIBLE_DEVICES=0  # GPU 선택
export OMP_NUM_THREADS=8       # CPU 스레드 수
export FINRL_DATA_DIR=./data   # 데이터 디렉토리
```

### 12.4 디렉토리 구조

```
FinFlow-rl/
├── finrl/
│   ├── agents/
│   │   └── irt/
│   │       ├── __init__.py
│   │       ├── irt_policy.py      # IRTPolicy, IRTActorWrapper
│   │       ├── bcell_actor.py     # BCellIRTActor (핵심 구현)
│   │       ├── irt_operator.py    # IRT Operator (OT + Rep)
│   │       └── t_cell.py          # TCellMinimal (위기 감지)
│   ├── meta/
│   │   ├── env_stock_trading/
│   │   │   ├── env_stocktrading.py
│   │   │   └── reward_functions.py
│   │   └── preprocessor/
│   └── evaluation/
│       ├── metrics.py
│       └── visualizer.py
├── scripts/
│   ├── train_irt.py               # 학습 스크립트
│   ├── evaluate.py                # 평가 스크립트
│   ├── compare_models.py          # 모델 비교
│   └── visualize_from_json.py     # 시각화
├── tests/
│   ├── test_irt_policy.py         # 단위 테스트
│   └── phase_gate.py              # Phase gate 검증
├── logs/
│   └── irt/
│       └── <timestamp>/
│           ├── irt_final.zip      # 최종 모델
│           ├── best_model.zip     # 최고 성능 모델
│           ├── tensorboard/       # TensorBoard 로그
│           └── env_meta.json      # 환경 메타데이터
├── docs/
│   ├── IRT.md                     # 본 문서
│   ├── CHANGELOG.md
│   └── SCRIPTS.md
├── requirements.txt
├── setup.py
└── README.md
```

### 12.5 최소 동작 예시

**학습**:

```bash
python scripts/train_irt.py --mode train --episodes 60 --reward-type adaptive_risk
```

**평가**:

```bash
python scripts/evaluate.py \
  --model logs/irt/<timestamp>/irt_final.zip \
  --method direct \
  --viz all \
  --xai-level full
```

**시각화**:

```bash
python scripts/visualize_from_json.py \
  --input-dir logs/irt/<timestamp>/evaluation_results \
  --html-index
```

### 12.6 라이센스 및 인용

**라이센스**: MIT License

**인용**:

```bibtex
@misc{finrl-irt-2025,
  title={IRT: Immune Replicator Transport for Crisis-Adaptive Portfolio Management},
  author={FinRL-IRT Team},
  year={2025},
  howpublished={\url{https://github.com/reo91004/FinFlow-rl}}
}
```

---

**마지막 업데이트**: 2025-10-16
**버전**: 1.0.0
**문의**: GitHub Issues 또는 Discussions
