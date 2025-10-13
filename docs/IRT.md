# IRT (Immune Replicator Transport) 설명서

## 📋 목차

1. [개요](#개요)
2. [내부 아키텍처](#내부-아키텍처)
3. [IRT의 핵심 메커니즘](#irt의-핵심-메커니즘)
4. [위기 감지 시스템](#위기-감지-시스템)
5. [적응형 메커니즘](#적응형-메커니즘)
6. [프로토타입 학습](#프로토타입-학습)
7. [다른 알고리즘과의 비교](#다른-알고리즘과의-비교)
8. [하이퍼파라미터 가이드](#하이퍼파라미터-가이드)
9. [사용 예시](#사용-예시)
10. [참고 문헌](#참고-문헌)
11. [성능 목표 및 기대 효과](#성능-목표-및-기대-효과)

---

## 개요

### IRT란 무엇인가?

IRT (Immune Replicator Transport)는 **면역학적 메커니즘에서 영감을 받은** 위기 적응형 포트폴리오 관리 알고리즘이다. 금융 시장의 정상 구간과 위기 구간에서 서로 다른 최적 전략이 필요하다는 인사이트를 바탕으로, 세 가지 핵심 메커니즘을 결합한다:

1. **Optimal Transport (OT)**: 현재 시장 상태와 전문가 전략 간의 구조적 매칭
2. **Replicator Dynamics**: 과거 성공 전략에 대한 시간 메모리
3. **T-Cell Crisis Detection**: 다중 신호 기반 실시간 위기 감지

### 핵심 공식

```
w_t = (1-α_c)·Replicator(w_{t-1}, f_t) + α_c·Transport(E_t, K, C_t)
```

**주요 특징**:

- `α_c`: **동적 혼합 비율** - 위기 상황에 따라 자동 조절
  - 평시(c≈0): α_c ≈ α_max (OT 증가, 탐색적 매칭)
  - 위기(c≈1): α_c ≈ α_min (Replicator 증가, 검증된 전략 선호)
- `w_t`: 프로토타입 혼합 가중치 [B, M]
- `f_t`: Critic Q-value 기반 프로토타입 적합도 [B, M]
- `E_t`: 에피토프(상태 인코딩) [B, m, D]
- `K`: 학습 가능한 프로토타입 키 [M, D]

### 왜 위기 적응에 효과적인가?

일반적인 강화학습 알고리즘은 **단일 정책**을 학습한다. 하지만 금융 시장은 **정상 구간**과 **위기 구간**에서 완전히 다른 특성을 보인다:

- **정상 구간**: 낮은 변동성, 예측 가능한 패턴, 분산 투자 유리
- **위기 구간**: 높은 변동성, 급격한 변화, 상관관계 붕괴, 방어적 전략 필요

IRT는 **M개의 전문가 프로토타입**을 학습하고, 현재 상황에 따라 동적으로 혼합한다:

```
시장 관찰 → T-Cell 다중 신호 위기 감지 (시장 통계 + DSR + CVaR)
         → 위기 레벨 산출 (바이어스 보정 + 히스테리시스)
         → α_c 동적 조절 (위기 시 Replicator 가중)
         → η 가열 (빠른 적응)
         → OT 비용 함수 조정 (면역학적 신호)
         → 최적 프로토타입 혼합 선택
         → 위기 적응형 포트폴리오 구성
```

---

## 내부 아키텍처

### 전체 시스템 구조

IRT는 Stable Baselines3의 SAC 알고리즘과 완전히 통합되어 작동한다. SAC의 Critic (Q-network)을 재사용하고 Actor만 IRT로 교체하는 구조다.

#### 고수준 아키텍처 (High-Level Architecture)

```
┌───────────────────────────────────────────────────────────────────────┐
│                          SAC Training Framework                        │
├───────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────┐         ┌─────────────────┐       ┌──────────────┐ │
│  │    Replay    │────────>│   IRT Policy    │<──────│ Twin Critics │ │
│  │    Buffer    │         │   (π_θ)         │       │   (Q_φ1, Q_φ2)│ │
│  │              │<────────│                 │       │              │ │
│  │  (s,a,r,s')  │  store  └────────┬────────┘       └──────┬───────┘ │
│  └──────────────┘                  │                       │         │
│                                    │ Q-values [B,M]        │         │
│                                    │<──────────────────────┘         │
│                                    │ Q(s,π_j(s))                     │
│                                    ▼                                 │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                      IRT Actor Network                          │ │
│  │                                                                 │ │
│  │  Input: s_t ∈ ℝ^d                                              │ │
│  │  Output: a_t ∈ Δ^n (Simplex)                                   │ │
│  │                                                                 │ │
│  │  ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐  │ │
│  │  │   Crisis    │    │  State       │    │   Expert         │  │ │
│  │  │   Detector  │───>│  Encoder     │    │   Strategies     │  │ │
│  │  │             │    │  (ϕ_s)       │    │   {θ_j}^M_j=1    │  │ │
│  │  │ c_t, h^c_t  │    │  h^s_t       │───>│  IRT Operator    │  │ │
│  │  └─────────────┘    └──────────────┘    │  (Ψ_IRT)         │  │ │
│  │                                          │                  │  │ │
│  │  Q-values Q_t ───────────────────────────> w_t = Ψ_IRT()   │  │ │
│  │  [M]                                     │  [M]             │  │ │
│  │                                          └────────┬─────────┘  │ │
│  │                                                   │             │ │
│  │  ┌────────────────────────────────────────────────▼──────────┐ │ │
│  │  │           Mixture Policy (ϕ_π)                            │ │ │
│  │  │                                                            │ │ │
│  │  │  α_j = MLP_j(θ_j)  [n]                                    │ │ │
│  │  │  α_t = Σ_j w_{t,j} · α_j  [n]                             │ │ │
│  │  │                                                            │ │ │
│  │  │  a_t ~ Dir(α_t)  or  a_t = softmax(α_t/τ)                │ │ │
│  │  └────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                    │                                 │
│                                    ▼                                 │
│                          Environment (s', r)                         │
│                                                                        │
└───────────────────────────────────────────────────────────────────────┘
```

#### IRT Actor 상세 아키텍처 (Detailed IRT Actor Architecture)

```
                    Input State s_t ∈ ℝ^d  (d=181)
                           │
      ┌────────────────────┼────────────────────┐
      │                    │                    │
      ▼                    ▼                    ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Market     │    │   State      │    │  Q-value        │
│  Features   │    │   Features   │    │  Estimation     │
│  Extraction │    │              │    │  (Twin Critics) │
│             │    │              │    │                 │
│  x^m ∈ ℝ^12 │    │  s_t ∈ ℝ^d   │    │  Q_j=Q(s,π_j(s))│
└──────┬──────┘    └──────┬───────┘    └────────┬────────┘
       │                  │                     │
       │                  │                     │ Q ∈ ℝ^M
       ▼                  ▼                     │
┌──────────────┐   ┌─────────────┐             │
│  Crisis      │   │  State      │             │
│  Detector    │   │  Encoder    │             │
│  (ϕ_c)       │   │  (ϕ_s)      │             │
│              │   │             │             │
│  h = MLP(x^m)│   │  h = MLP(s) │             │
│  z, h^c = h  │   │  h^s = h    │             │
│              │   │  [m×D]      │             │
└──────┬───────┘   └──────┬──────┘             │
       │                  │                     │
       │ ┌────────────────┘                     │
       │ │                                      │
       ▼ ▼                                      │
  ┌─────────────────────────────────────┐       │
  │    Multi-Signal Crisis Assessment   │       │
  │    ────────────────────────────     │       │
  │                                     │       │
  │  Signal Extraction:                │       │
  │    - Market signal: z [4]          │       │
  │    - Sharpe gradient: Δs           │       │
  │    - CVaR signal: c_v              │       │
  │                                     │       │
  │  Z-Score Normalization:            │       │
  │    z_i = (x_i - μ_i)/σ_i  (EMA)   │       │
  │                                     │       │
  │  Weighted Fusion:                  │       │
  │    c_r = w_m·σ(k_m·z_m)            │       │
  │        + w_s·σ(k_s·z_s)            │       │
  │        + w_c·σ(k_c·z_c)            │       │
  │                                     │       │
  │  Bias/Temperature Correction:      │       │
  │    c̃ = σ((c_r - b)/T)              │       │
  │                                     │       │
  │  Target-Driven Regularization:     │       │
  │    c = c̃ + λ_g·(c_target - c̃)     │       │
  │                                     │       │
  │  Output: c_t ∈ [0,1], h^c_t ∈ ℝ^D  │       │
  └────────────────┬────────────────────┘       │
                   │                            │
                   │                            │
       ┌───────────┴────────────────────────────┘
       │
       ▼
  ┌────────────────────────────────────────────────┐
  │          IRT Operator (Ψ_IRT)                  │
  │          ────────────────────                  │
  │                                                │
  │  Input:  h^s_t [m×D], {θ_j} [M×D], h^c_t [D], │
  │          w_{t-1} [M], Q_t [M], c_t, Δs_t      │
  │                                                │
  │  ┌──────────────────┐  ┌────────────────────┐ │
  │  │ Optimal          │  │ Replicator         │ │
  │  │ Transport        │  │ Dynamics           │ │
  │  │ ─────────        │  │ ────────           │ │
  │  │                  │  │                    │ │
  │  │ Distance Matrix: │  │ Adaptive LR:       │ │
  │  │  C = d_M(h^s,θ)  │  │  η = η_0+η_1·c    │ │
  │  │    - γ⟨h^s,h^c⟩  │  │                    │ │
  │  │    + penalties   │  │ Advantage:         │ │
  │  │                  │  │  A = Q - w^T Q    │ │
  │  │ Sinkhorn OT:     │  │                    │ │
  │  │  P* = S(C,ε)     │  │ Update:            │ │
  │  │                  │  │  w̃ ∝ w·exp(η·A)   │ │
  │  │ Marginal:        │  │  w_r=softmax(w̃/τ) │ │
  │  │  w_o = P*^T 1_m  │  │                    │ │
  │  │                  │  │                    │ │
  │  └────────┬─────────┘  └─────────┬──────────┘ │
  │           │                      │            │
  │           │  ┌───────────────────┘            │
  │           │  │                                │
  │           ▼  ▼                                │
  │  ┌────────────────────────┐                   │
  │  │  Dynamic α Schedule    │                   │
  │  │  ──────────────────    │                   │
  │  │                        │                   │
  │  │  α(c) = α_max + ...    │                   │
  │  │  α'(Δs) = α·(1+...)    │                   │
  │  │                        │                   │
  │  │  w_t = (1-α')·w_r      │                   │
  │  │      + α'·w_o          │                   │
  │  └────────────────────────┘                   │
  │                                                │
  │  Output: w_t ∈ Δ^M (Expert weights)           │
  └──────────────────────┬─────────────────────────┘
                         │
                         │ w_t [M]
                         ▼
  ┌────────────────────────────────────────────────┐
  │         Mixture Policy Network (ϕ_π)           │
  │         ────────────────────────────           │
  │                                                │
  │  Expert Policy Heads: {MLP_j}^M_j=1           │
  │                                                │
  │  ∀j: α_j = MLP_j(θ_j) ∈ ℝ^n                   │
  │       α_j = clamp(tanh(·)·7.5+2.5, 0.01)      │
  │                                                │
  │  Weighted Aggregation:                        │
  │    α_t = Σ^M_j=1 w_{t,j} · α_j               │
  │                                                │
  │  Action Sampling:                             │
  │    Training:   a_t ~ Dirichlet(α_t)           │
  │    Inference:  a_t = softmax(α_t/τ_a)         │
  │                                                │
  │  Simplex Projection: a_t ∈ Δ^n               │
  └──────────────────────┬─────────────────────────┘
                         │
                         ▼
                    Output Action a_t ∈ Δ^n  (n=30)
                    Portfolio Weights
```

#### 데이터 플로우 및 차원

| 모듈 | 입력 | 출력 | 설명 |
|------|------|------|------|
| **Market Features** | s_t [d] | x^m [12] | 시장 통계 + 기술 지표 |
| **Crisis Detector (ϕ_c)** | x^m [12] | z [4], h^c [D] | 위기 신호 벡터, 위기 임베딩 |
| **Multi-Signal Assessment** | z, Δs, c_v | c [1] | 다중 신호 융합 → 위기 레벨 |
| **State Encoder (ϕ_s)** | s_t [d] | h^s [m×D] | 상태 → 다중 표현 토큰 |
| **Expert Strategies** | - | {θ_j} [M×D] | 학습 가능한 전문가 파라미터 |
| **Q-value Estimation** | s_t, {θ_j} | Q [M] | Twin Critic Q-networks |
| **IRT Operator (Ψ_IRT)** | h^s, {θ_j}, h^c, w_prev, Q, c, Δs | w [M] | OT + Replicator 결합 |
| **Expert Policy Heads {MLP_j}** | θ_j [D] | α_j [n] | 전문가 → Concentration |
| **Mixture Policy (ϕ_π)** | w [M], {α_j} [M×n] | a_t [n] | 최종 포트폴리오 |

**표기법**:
- d: 상태 차원 (181)
- n: 행동 차원 (30, 자산 수)
- m: 상태 표현 토큰 수 (6)
- M: 전문가 수 (8)
- D: 임베딩 차원 (128)
- Δ^k: k-simplex (합이 1인 양수 벡터)
- [k]: k-차원 벡터
- [k×l]: k×l 행렬

---

### 용어 매핑: 논문 ↔ 코드베이스

본 문서에서는 논문 작성을 위해 formal한 용어를 사용하지만, 실제 코드베이스에서는 면역학적 비유를 사용한다. 다음은 용어 매핑표다:

| 논문 용어 (Formal) | 코드베이스 용어 (Metaphorical) | 구현 위치 |
|-------------------|-------------------------------|----------|
| **IRT Operator** | IRT Operator | `finrl/agents/irt/irt_operator.py` |
| **Crisis Detector (ϕ_c)** | T-Cell Network | `finrl/agents/irt/t_cell.py:TCellMinimal` |
| **State Encoder (ϕ_s)** | Epitope Encoder | `finrl/agents/irt/bcell_actor.py:epitope_encoder` |
| **Expert Strategies {θ_j}** | Prototype Keys | `finrl/agents/irt/bcell_actor.py:proto_keys` |
| **Expert Policy Heads {MLP_j}** | Decoders | `finrl/agents/irt/bcell_actor.py:decoders` |
| **Q-values (Q_t)** | Fitness | `finrl/agents/irt/irt_policy.py:_compute_fitness()` |
| **State Representation (h^s)** | Epitope Tokens (E) | `bcell_actor.py:L439` |
| **Crisis Embedding (h^c)** | Danger Embedding | `bcell_actor.py:L324` |
| **Expert Weights (w_t)** | Prototype Weights | `irt_operator.py:L312` |
| **Distance Matrix (C)** | Immunological Cost Matrix | `irt_operator.py:L171-222` |
| **Multi-Signal Assessment** | T-Cell Multi-Signal Detection | `bcell_actor.py:L292-434` |
| **Z-Score Normalization** | EMA Statistics | `bcell_actor.py:L331-347` |
| **Bias/Temperature Correction** | Crisis Bias/Temperature | `bcell_actor.py:L389-407` |
| **Target-Driven Regularization** | T-Cell Guard | `bcell_actor.py:L409-415` |
| **Mixture Policy (ϕ_π)** | Dirichlet Mixture Policy | `bcell_actor.py:L466-498` |
| **Adaptive Learning Rate (η)** | Crisis Heating | `irt_operator.py:L265-267` |
| **Dynamic α Schedule** | Dynamic OT-Replicator Mixing | `irt_operator.py:L288-303` |

**주요 용어 설명**:

1. **Crisis Detector → T-Cell**
   - 논문: 위기 감지 신경망
   - 코드: T-Cell (면역 세포의 비유)
   - 이유: 면역계의 T-Cell이 병원체를 감지하는 것처럼 시장 위기를 감지

2. **State Encoder → Epitope Encoder**
   - 논문: 상태를 다중 토큰으로 인코딩
   - 코드: Epitope (항원 결정기의 비유)
   - 이유: 면역계의 항원-항체 결합처럼 상태-전략 매칭

3. **Expert Strategies → Prototypes**
   - 논문: 학습 가능한 전문가 전략 파라미터
   - 코드: Prototype Keys (원형의 비유)
   - 이유: 각 전문가가 특정 시장 조건의 프로토타입 전략

4. **Q-values → Fitness**
   - 논문: Critic의 Q-value 추정
   - 코드: Fitness (진화론적 적합도)
   - 이유: Replicator Dynamics가 진화 게임 이론에서 유래

5. **Distance Matrix → Immunological Cost**
   - 논문: 상태-전략 간 거리 행렬
   - 코드: Immunological Cost (면역학적 비용)
   - 이유: 면역 반응의 비용 개념을 OT 비용으로 해석

**코드 예시**:

```python
# 논문: Crisis Detector
# 코드: T-Cell Network
self.t_cell = TCellMinimal(in_dim=market_feature_dim, emb_dim=emb_dim)
# 위치: bcell_actor.py:196-199

# 논문: State Encoder
# 코드: Epitope Encoder
self.epitope_encoder = nn.Sequential(...)
# 위치: bcell_actor.py:147-153

# 논문: Expert Strategies
# 코드: Prototype Keys
self.proto_keys = nn.Parameter(torch.randn(M_proto, emb_dim))
# 위치: bcell_actor.py:156-159

# 논문: Q-values
# 코드: Fitness
fitness = self._compute_fitness(obs)
# 위치: irt_policy.py:154, 183

# 논문: IRT Operator
# 코드: IRT (동일)
w, P, irt_debug = self.irt(E, K, danger_embed, w_prev, fitness, ...)
# 위치: bcell_actor.py:455-464
```

이러한 비유적 명명은 알고리즘의 생물학적 영감을 강조하기 위함이나, 논문에서는 formal한 용어를 사용하는 것이 표준적이다.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    SAC Training Loop                                     │
│                                                                                          │
│  ┌──────────────────┐         ┌────────────────────────────────────┐                   │
│  │  Replay Buffer   │         │      IRTPolicy (SACPolicy)         │                   │
│  │  ────────────    │         │  ┌──────────────────────────────┐ │                   │
│  │  (s,a,r,s',d)    │◄────────┼──│  Features Extractor          │ │                   │
│  │  100k samples    │  store  │  │  (FlattenExtractor)          │ │                   │
│  └──────────────────┘         │  │  state[181] → features[181]  │ │                   │
│         │                     │  └──────────────────────────────┘ │                   │
│         │ sample batch        │              │                     │                   │
│         │ (256)               │              ├─────────────────────┼──────────────┐    │
│         ▼                     │              ▼                     │              │    │
│  ┌──────────────────┐         │  ┌──────────────────────────────┐ │              │    │
│  │ Train Step       │         │  │   IRTActorWrapper (Actor)     │ │              │    │
│  │ ──────────       │         │  │   ───────────────────────     │ │              │    │
│  │ 1. Critic Update │◄────────┼──│                               │ │              │    │
│  │ 2. Actor Update  │         │  │  ┌─────────────────────────┐ │ │              │    │
│  │ 3. Target Update │         │  │  │ _compute_fitness(obs)   │ │ │              │    │
│  └──────────────────┘         │  │  │ ─────────────────────   │ │ │              │    │
│         │                     │  │  │                         │ │ │       ┌──────▼────▼─────┐
│         │                     │  │  │  For j in M:           │ │ │       │  Twin Critic    │
│         │                     │  │  │    conc_j = decoder_j()├─┼─┼───────►  Q-networks     │
│         │                     │  │  │    a_j = softmax(conc) │ │ │       │  ──────────     │
│         │                     │  │  │    fitness[j] = min(   ├─┼─┼───────►  Q1(s,a), Q2(s,a)│
│         │                     │  │  │      Q1(obs,a_j),      │ │ │       └──────────────────┘
│         │                     │  │  │      Q2(obs,a_j))      │ │ │              │
│         │                     │  │  │                         │ │ │              │ Q-values
│         │                     │  │  └─────────────────────────┘ │ │              │
│         │                     │  │              │                │ │              │
│         │                     │  │              ▼ fitness[B,M]   │ │              │
│         │                     │  │  ┌─────────────────────────┐ │ │              │
│         │                     │  │  │  BCellIRTActor.forward()│ │ │              │
│         │                     │  │  │  ───────────────────────│ │ │              │
│         │                     │  │  │  (state, fitness, det)  │ │ │              │
│         │                     │  │  └────────────┬────────────┘ │ │              │
│         │                     │  │               │              │ │              │
│         │                     │  │               ▼              │ │              │
│         │                     │  │  ┌──────────────────────────────────────────┐ │
│         │                     │  │  │         IRT Core (6 Steps)               │ │
│         │                     │  │  │         ──────────────────               │ │
│         │                     │  │  │                                          │ │
│         │                     │  │  │  ┌────────────────────────────────────┐ │ │
│         │                     │  │  │  │ Step 1: T-Cell Crisis Detection    │ │ │
│         │                     │  │  │  │ ───────────────────────────────    │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  Market Features (12-dim)          │ │ │
│         │                     │  │  │  │  ├─ balance, price_mean/std, ...   │ │ │
│         │                     │  │  │  │  └─ macd, boll, rsi, cci, ...      │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  TCellMinimal(market_feat)         │ │ │
│         │                     │  │  │  │    → crisis_affine, danger_embed   │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  DSR/CVaR Signals                  │ │ │
│         │                     │  │  │  │  ├─ delta_sharpe (state[-2])       │ │ │
│         │                     │  │  │  │  └─ cvar (state[-1])               │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  EMA Normalization (z-score)       │ │ │
│         │                     │  │  │  │  ├─ base_z, sharpe_z, cvar_z       │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  Multi-Signal Fusion               │ │ │
│         │                     │  │  │  │    crisis_raw = w_r·sigmoid(k_b·z) │ │ │
│         │                     │  │  │  │               + w_s·sigmoid(-k_s·z)│ │ │
│         │                     │  │  │  │               + w_c·sigmoid(k_c·z) │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  Bias/Temperature Correction       │ │ │
│         │                     │  │  │  │    affine = (raw - bias) / temp    │ │ │
│         │                     │  │  │  │    crisis_pre = sigmoid(affine)    │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  T-Cell Guard                      │ │ │
│         │                     │  │  │  │    crisis = pre + rate·(0.5 - pre) │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  Hysteresis Regime                 │ │ │
│         │                     │  │  │  │    → crisis_level, crisis_regime   │ │ │
│         │                     │  │  │  └────────────────────────────────────┘ │ │
│         │                     │  │  │                │                        │ │
│         │                     │  │  │                ▼                        │ │
│         │                     │  │  │  ┌────────────────────────────────────┐ │ │
│         │                     │  │  │  │ Step 2: Epitope Encoding           │ │ │
│         │                     │  │  │  │ ────────────────────               │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  epitope_encoder(state[181])       │ │ │
│         │                     │  │  │  │    → E [B, m=6, D=128]             │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  Multi-Token Representation:       │ │ │
│         │                     │  │  │  │  - Token 1~6: Different aspects    │ │ │
│         │                     │  │  │  │    of market state                 │ │ │
│         │                     │  │  │  └────────────────────────────────────┘ │ │
│         │                     │  │  │                │                        │ │
│         │                     │  │  │                ▼                        │ │
│         │                     │  │  │  ┌────────────────────────────────────┐ │ │
│         │                     │  │  │  │ Step 3: Prototype Keys             │ │ │
│         │                     │  │  │  │ ──────────────────                 │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  proto_keys [M=8, D=128]           │ │ │
│         │                     │  │  │  │    → K [B, M=8, D=128]             │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  Learnable Expert Strategies:      │ │ │
│         │                     │  │  │  │  - Proto 0~7: Specialized for      │ │ │
│         │                     │  │  │  │    different market conditions     │ │ │
│         │                     │  │  │  └────────────────────────────────────┘ │ │
│         │                     │  │  │                │                        │ │
│         │                     │  │  │                ▼                        │ │
│         │                     │  │  │  ┌────────────────────────────────────┐ │ │
│         │                     │  │  │  │ Step 4: IRT Operator               │ │ │
│         │                     │  │  │  │ ────────────────                   │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  IRT(E, K, danger, w_prev,         │ │ │
│         │                     │  │  │  │      fitness, crisis, delta_sharpe)│ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  ┌──────────────────────────────┐ │ │ │
│         │                     │  │  │  │  │ [A] Optimal Transport Path   │ │ │ │
│         │                     │  │  │  │  │ ─────────────────────────    │ │ │ │
│         │                     │  │  │  │  │                              │ │ │ │
│         │                     │  │  │  │  │ Cost Matrix (Immunological): │ │ │ │
│         │                     │  │  │  │  │   C = d_M(E,K)               │ │ │ │
│         │                     │  │  │  │  │     - γ·<E, danger>          │ │ │ │
│         │                     │  │  │  │  │     + λ·tolerance_penalty    │ │ │ │
│         │                     │  │  │  │  │     + ρ·checkpoint_penalty   │ │ │ │
│         │                     │  │  │  │  │                              │ │ │ │
│         │                     │  │  │  │  │ Sinkhorn Algorithm:          │ │ │ │
│         │                     │  │  │  │  │   P* = argmin <P,C> + ε·KL   │ │ │ │
│         │                     │  │  │  │  │                              │ │ │ │
│         │                     │  │  │  │  │ OT Margin:                   │ │ │ │
│         │                     │  │  │  │  │   w_ot = Σ_i P*(i,j)         │ │ │ │
│         │                     │  │  │  │  │          [B, M]              │ │ │ │
│         │                     │  │  │  │  └──────────────────────────────┘ │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  ┌──────────────────────────────┐ │ │ │
│         │                     │  │  │  │  │ [B] Replicator Dynamics Path │ │ │ │
│         │                     │  │  │  │  │ ────────────────────────────│ │ │ │
│         │                     │  │  │  │  │                              │ │ │ │
│         │                     │  │  │  │  │ Crisis Heating:              │ │ │ │
│         │                     │  │  │  │  │   η(c) = η_0 + η_1·c         │ │ │ │
│         │                     │  │  │  │  │                              │ │ │ │
│         │                     │  │  │  │  │ Advantage:                   │ │ │ │
│         │                     │  │  │  │  │   baseline = Σ w·fitness     │ │ │ │
│         │                     │  │  │  │  │   A = fitness - baseline     │ │ │ │
│         │                     │  │  │  │  │                              │ │ │ │
│         │                     │  │  │  │  │ Replicator Equation:         │ │ │ │
│         │                     │  │  │  │  │   log_w̃ = log(w_prev)        │ │ │ │
│         │                     │  │  │  │  │          + η·A - r_penalty   │ │ │ │
│         │                     │  │  │  │  │                              │ │ │ │
│         │                     │  │  │  │  │ Temperature Softmax:         │ │ │ │
│         │                     │  │  │  │  │   w_rep = softmax(log_w̃/τ)   │ │ │ │
│         │                     │  │  │  │  │           [B, M]             │ │ │ │
│         │                     │  │  │  │  └──────────────────────────────┘ │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  ┌──────────────────────────────┐ │ │ │
│         │                     │  │  │  │  │ [C] Dynamic Mixing           │ │ │ │
│         │                     │  │  │  │  │ ──────────────               │ │ │ │
│         │                     │  │  │  │  │                              │ │ │ │
│         │                     │  │  │  │  │ Dynamic α(c):                │ │ │ │
│         │                     │  │  │  │  │   α_c = α_max + (α_min-α_max)│ │ │ │
│         │                     │  │  │  │  │         ·(1-cos(πc))/2       │ │ │ │
│         │                     │  │  │  │  │                              │ │ │ │
│         │                     │  │  │  │  │ Sharpe Gradient Feedback:    │ │ │ │
│         │                     │  │  │  │  │   α'= α·(1+0.6·tanh(Δsharpe))│ │ │ │
│         │                     │  │  │  │  │      + 0.07·tanh(Δsharpe)    │ │ │ │
│         │                     │  │  │  │  │   α_c = clamp(α', min, max)  │ │ │ │
│         │                     │  │  │  │  │                              │ │ │ │
│         │                     │  │  │  │  │ Fusion:                      │ │ │ │
│         │                     │  │  │  │  │   w = (1-α_c)·w_rep          │ │ │ │
│         │                     │  │  │  │  │     + α_c·w_ot               │ │ │ │
│         │                     │  │  │  │  │       [B, M]                 │ │ │ │
│         │                     │  │  │  │  └──────────────────────────────┘ │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  → w, P, debug_info                │ │ │
│         │                     │  │  │  └────────────────────────────────────┘ │ │
│         │                     │  │  │                │                        │ │
│         │                     │  │  │                ▼                        │ │
│         │                     │  │  │  ┌────────────────────────────────────┐ │ │
│         │                     │  │  │  │ Step 5: Dirichlet Mixing           │ │ │
│         │                     │  │  │  │ ────────────────────               │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  For each prototype j:             │ │ │
│         │                     │  │  │  │    decoder_j(K[:,j,:])             │ │ │
│         │                     │  │  │  │      → conc_j [B, action_dim=30]   │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  Tanh → Concentration:             │ │ │
│         │                     │  │  │  │    conc = clamp(tanh·7.5+2.5, 0.01)│ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  IRT Weighted Mixing:              │ │ │
│         │                     │  │  │  │    mixed_conc = Σ_j w_j·conc_j     │ │ │
│         │                     │  │  │  │                 [B, 30]            │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  if deterministic:                 │ │ │
│         │                     │  │  │  │    action = softmax(conc/temp)     │ │ │
│         │                     │  │  │  │  else:                             │ │ │
│         │                     │  │  │  │    action ~ Dirichlet(clamp(conc)) │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  Simplex Projection:               │ │ │
│         │                     │  │  │  │    action = action / Σ_i action_i  │ │ │
│         │                     │  │  │  └────────────────────────────────────┘ │ │
│         │                     │  │  │                │                        │ │
│         │                     │  │  │                ▼                        │ │
│         │                     │  │  │  ┌────────────────────────────────────┐ │ │
│         │                     │  │  │  │ Step 6: EMA Update                 │ │ │
│         │                     │  │  │  │ ──────────────                     │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │  if training:                      │ │ │
│         │                     │  │  │  │    w_prev = β·w_prev               │ │ │
│         │                     │  │  │  │           + (1-β)·w.mean(dim=0)    │ │ │
│         │                     │  │  │  │                                    │ │ │
│         │                     │  │  │  │    # Bias/Temperature adaptation   │ │ │
│         │                     │  │  │  │    bias += η_b·(p_hat - p_star)    │ │ │
│         │                     │  │  │  │    temp *= (1 + η_T·(p_hat-p_star))│ │ │
│         │                     │  │  │  └────────────────────────────────────┘ │ │
│         │                     │  │  │                                          │ │
│         │                     │  │  └──────────────────────────────────────────┘ │
│         │                     │  │                   │                            │
│         │                     │  │                   ▼                            │
│         │                     │  │         action [B, 30], info (dict)            │
│         │                     │  └────────────────────────────────────────────────┘
│         │                     │                     │                               │
│         │                     └─────────────────────┼───────────────────────────────┘
│         │                                           │                               
│         │                                           ▼                               
│         │                                    action [B, 30]                         
│         │                                           │                               
│         ▼                                           │                               
│  ┌─────────────────────────────────────────────────▼────────────────┐              
│  │                  Environment Interaction                          │              
│  │  ────────────────────────────────────                             │              
│  │                                                                    │              
│  │  PortfolioOptimizationEnv:                                        │              
│  │    - Apply action (portfolio weights)                             │              
│  │    - Calculate reward (Sharpe/DSR/CVaR)                           │              
│  │    - Update state (prices, balance, shares, indicators)           │              
│  │    - Return (s', r, done, info)                                   │              
│  │                                                                    │              
│  └────────────────────────────────────────────────────────────────────┘              
│                                           │                                         
│                                           │ (s', r, done, info)                     
│                                           ▼                                         
│                                   Back to Replay Buffer                             
│                                                                                     
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**계층별 데이터 흐름**:

1. **Environment → Policy**: `state [B, 181]`
   - 181-dim: balance(1) + prices(30) + shares(30) + tech_indicators(240×8) + DSR(1) + CVaR(1)

2. **Policy → Actor**: `obs [B, 181]`

3. **Actor → Critic**: `a_j [B, 30]` (각 프로토타입 샘플)
   - **Critic → Actor**: `fitness [B, M=8]` (Q-values)

4. **Actor Internal**:
   - Market features `[B, 12]` → T-Cell → `crisis_level [B, 1]`, `danger [B, D]`
   - State `[B, 181]` → Epitope Encoder → `E [B, m=6, D=128]`
   - Proto keys `[M=8, D]` → expand → `K [B, M, D]`
   - IRT → `w [B, M]` (프로토타입 믹싱 가중치)
   - Decoders → `concentrations [B, M, 30]`
   - Mix → `action [B, 30]` (포트폴리오 가중치)

5. **Actor → Environment**: `action [B, 30]`

6. **Environment → Buffer**: `(s, a, r, s', done)`

### Stable Baselines3 통합 흐름

**학습 시 (Training)**:

```
SAC.train()
  │
  ├─> collect_rollouts() → buffer에 (s,a,r,s') 저장
  │
  └─> train_step()
       │
       ├─> Sample batch from buffer
       │
       ├─> Critic 업데이트
       │    └─> Q-loss = MSE(Q(s,a), r + γ·Q_target(s', a'))
       │
       └─> Actor 업데이트
            │
            └─> policy.actor.action_log_prob(obs) 호출
                 │
                 └─> IRTActorWrapper.action_log_prob(obs)
                      │
                      ├─> obs.float()  # dtype 변환 (float64 → float32)
                      │
                      ├─> _compute_fitness(obs)  # Critic Q-value로 fitness 계산
                      │    │
                      │    └─> 각 프로토타입의 샘플 행동에 대해 Q(s,a) 계산
                      │         fitness[j] = min(Q1(s,aj), Q2(s,aj))
                      │
                      └─> BCellIRTActor(obs, fitness, deterministic=False)
                           │
                           └─> (action, info) 반환
                                └─> log_prob = Dirichlet(mixed_conc).log_prob(action)
```

**평가 시 (Evaluation)**:

```
model.predict(obs, deterministic=True)
  │
  └─> IRTActorWrapper.forward(obs, deterministic=True)
       │
       ├─> obs.float()
       │
       ├─> _compute_fitness(obs)  # 동일한 Critic 기반 fitness
       │
       └─> BCellIRTActor(obs, fitness, deterministic=True)
            │
            └─> action = softmax(mixed_conc / temp)  # 결정적 행동
```

### BCellIRTActor: IRT 핵심 구현

BCellIRTActor는 IRT 알고리즘의 실제 구현체로, 6단계 프로세스를 거쳐 행동을 생성한다:

```
BCellIRTActor.forward(state, fitness, deterministic)
  │
  ├─> Step 1: T-Cell 다중 신호 위기 감지
  │    │
  │    ├─> Market features 추출 (12차원)
  │    │    ├─ 시장 통계 (4개): balance, price_mean, price_std, cash_ratio
  │    │    └─ Tech indicators (8개): macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, sma_30, sma_60
  │    │
  │    ├─> TCellMinimal(market_features)
  │    │    └─> crisis_affine, crisis_base_sigmoid, danger_embed
  │    │
  │    ├─> DSR/CVaR 신호 추출 (state 마지막 2차원)
  │    │    ├─ delta_sharpe: Sharpe ratio gradient (DSR bonus)
  │    │    └─ cvar: Conditional Value at Risk
  │    │
  │    ├─> EMA 정규화 (각 신호별 z-score)
  │    │    ├─ crisis_base_z = (crisis_affine - μ_base) / σ_base
  │    │    ├─ sharpe_z = (delta_sharpe - μ_sharpe) / σ_sharpe
  │    │    └─ cvar_z = (cvar - μ_cvar) / σ_cvar
  │    │
  │    ├─> 다중 신호 결합
  │    │    crisis_raw = w_r·sigmoid(k_b·base_z)
  │    │                + w_s·sigmoid(-k_s·sharpe_z)  # 음수: 하락 시 위기
  │    │                + w_c·sigmoid(k_c·cvar_z)
  │    │
  │    ├─> 바이어스/온도 보정 (EMA 적응)
  │    │    crisis_affine = (crisis_raw - bias) / temperature
  │    │    crisis_level_pre_guard = sigmoid(crisis_affine)
  │    │    
  │    │    # 학습 중: bias·temperature 자동 조정 (목표 crisis_pct = p_star)
  │    │    if training:
  │    │        bias += η_b · (crisis_pct - p_star)
  │    │        temperature *= (1 + η_T · (crisis_pct - p_star))
  │    │
  │    ├─> T-Cell Guard (목표 crisis_regime_pct 유도)
  │    │    crisis_level = crisis_pre_guard + guard_rate · (target - crisis_pre_guard)
  │    │
  │    └─> 히스테리시스 기반 레짐 전환
  │         if prev_regime == crisis:
  │             regime = (crisis_level < hysteresis_down) ? normal : crisis
  │         else:
  │             regime = (crisis_level > hysteresis_up) ? crisis : normal
  │
  ├─> Step 2: 에피토프 인코딩
  │    └─> E = epitope_encoder(state).view(B, m, D)  # [B, m=6, D=128]
  │
  ├─> Step 3: 프로토타입 확장
  │    └─> K = proto_keys.expand(B, -1, -1)  # [M=8, D] → [B, M, D]
  │
  ├─> Step 4: IRT 연산
  │    └─> w, P, debug = IRT(E, K, danger, w_prev, fitness, crisis_level, delta_sharpe)
  │         │
  │         ├─> [OT Path] 구조적 매칭
  │         │    ├─ 면역학적 비용 행렬 계산
  │         │    │   C = mahalanobis(E,K) - γ·<E,danger>
  │         │    │       + λ·tolerance_penalty + ρ·checkpoint_penalty
  │         │    │
  │         │    └─ Sinkhorn 알고리즘
  │         │         P* = argmin_P <P,C> + ε·KL(P||uv^T)
  │         │         w_ot = Σ_i P*(i,j)  # [B, M]
  │         │
  │         ├─> [Replicator Path] 시간 메모리
  │         │    ├─ 위기 가열: η(c) = η_0 + η_1·c
  │         │    │
  │         │    ├─ Advantage 계산
  │         │    │   A_j = fitness_j - Σ_j w_prev_j·fitness_j
  │         │    │
  │         │    └─ Replicator 방정식
  │         │         w_rep = softmax(log(w_prev) + η·A, τ=replicator_temp)
  │         │
  │         ├─> [Dynamic Mixing] 위기 적응형 α
  │         │    ├─ 동적 α 계산
  │         │    │   α_c(c) = α_max + (α_min - α_max)·(1 - cos(πc))/2
  │         │    │   c=0 (평시) → α_c=α_max (OT 증가)
  │         │    │   c=1 (위기) → α_c=α_min (Replicator 증가)
  │         │    │
  │         │    ├─ Sharpe gradient feedback (Phase C)
  │         │    │   α_c' = α_c·(1 + 0.6·tanh(delta_sharpe)) + 0.07·tanh(delta_sharpe)
  │         │    │   delta_sharpe > 0 → α_c 증가 (OT 탐색)
  │         │    │   delta_sharpe < 0 → α_c 감소 (Rep 보수적)
  │         │    │
  │         │    └─ Clamp: α_c ∈ [α_min, α_max]
  │         │
  │         └─> [Fusion] 이중 경로 결합
  │              w = (1 - α_c)·w_rep + α_c·w_ot  # [B, M]
  │
  ├─> Step 5: Dirichlet 정책 (프로토타입 혼합)
  │    │
  │    ├─> 각 프로토타입의 concentration 계산
  │    │    conc_j = decoder_j(K[:,j,:])  # [B, action_dim]
  │    │    # Tanh output [-1,1] → [0.01, ~10] 변환
  │    │    conc_j = clamp(conc_j * 7.5 + 2.5, min=0.01)
  │    │
  │    ├─> IRT 가중치로 혼합
  │    │    mixed_conc = Σ_j w_j · conc_j  # [B, action_dim]
  │    │
  │    └─> 행동 생성
  │         if deterministic:
  │             action = softmax(mixed_conc / action_temp)  # 결정적
  │         else:
  │             action = Dirichlet(clamp(mixed_conc, min, max)).sample()  # 확률적
  │
  └─> Step 6: EMA 업데이트 (시간 메모리)
       if training:
           w_prev = β·w_prev + (1-β)·w.mean(dim=0)
```

### 모듈별 역할

#### 1. IRTPolicy (SACPolicy 상속)

**역할**: SB3의 SAC와 IRT Actor를 연결하는 정책 인터페이스.

**핵심 파라미터**:
- IRT 구조: `emb_dim=128`, `m_tokens=6`, `M_proto=8`
- 동적 혼합: `alpha_min=0.08`, `alpha_max=0.45`
- 위기 신호: `w_r=0.55`, `w_s=-0.25`, `w_c=0.20`
- T-Cell 가드: `crisis_target=0.5`, `crisis_guard_rate_init=0.30`
- Replicator: `eta_0=0.05`, `eta_1=0.12`, `replicator_temp=1.4`
- OT: `eps=0.03`, `gamma=0.85`

**주요 메서드**:
- `make_actor()`: BCellIRTActor를 IRTActorWrapper로 감싸서 생성
- `get_irt_info()`: 마지막 forward의 IRT 디버그 정보 반환 (평가/시각화용)
- `_get_constructor_parameters()`: 체크포인트 저장용 파라미터

#### 2. IRTActorWrapper (Actor 인터페이스)

**역할**: BCellIRTActor를 SAC가 기대하는 Actor 인터페이스로 wrapping한다.

**핵심 설계**:
- SAC는 `actor(obs)` 시그니처를 기대하지만, IRT는 `(state, fitness)`가 필요
- Wrapper가 Critic Q-network로 fitness를 계산하여 IRT에 전달
- `forward()`와 `action_log_prob()`이 동일한 fitness 계산 로직 공유

**주요 메서드**:

```python
def _compute_fitness(self, obs):
    """Critic Q-value 기반 프로토타입 fitness 계산 (공통 helper)"""
    for j in range(M):
        # 프로토타입 j의 샘플 행동 생성
        conc_j = decoders[j](proto_keys[j])
        a_j = softmax(conc_j)  # Mode approximation
        
        # Twin Q-network 중 최소값 (conservative)
        q1, q2 = critic(obs, a_j)
        fitness[j] = min(q1, q2)
    return fitness

def forward(self, obs, deterministic):
    """평가 시 행동 생성"""
    obs = obs.float()  # dtype 안정성
    fitness = self._compute_fitness(obs)
    action, info = self.irt_actor(obs, fitness, deterministic)
    self._last_irt_info = info  # 시각화용 저장
    return action

def action_log_prob(self, obs):
    """학습 시 행동 + log_prob 계산"""
    obs = obs.float()
    fitness = self._compute_fitness(obs)  # 동일한 helper
    action, info = self.irt_actor(obs, fitness, deterministic=False)
    
    # Dirichlet log_prob 계산 (info에서 concentration 재사용)
    dist = Dirichlet(info['mixed_conc_clamped'])
    log_prob = dist.log_prob(action)
    
    self._last_irt_info = info
    return action, log_prob
```

**설계 이점**:
- ✅ **일관성**: train/eval 모두 동일한 fitness 계산
- ✅ **효율성**: IRT forward를 한 번만 호출 (EMA 메모리 보존)
- ✅ **안정성**: dtype 변환 + conservative Q estimate

#### 3. BCellIRTActor (IRT 구현)

**역할**: IRT 알고리즘의 핵심 구현체.

**아키텍처 구성요소**:

| 컴포넌트 | 타입 | 형상 | 역할 |
|---------|------|------|------|
| `epitope_encoder` | nn.Sequential | state → [B,m·D] | 상태를 m개 토큰으로 인코딩 |
| `proto_keys` | nn.Parameter | [M, D] | M개 학습 가능한 프로토타입 키 |
| `decoders` | nn.ModuleList[M] | [D] → [action_dim] | 프로토타입별 Dirichlet 정책 |
| `irt` | IRT | - | OT + Replicator 연산자 |
| `t_cell` | TCellMinimal | [market_dim] → crisis | 위기 감지 |
| `w_prev` | buffer | [1, M] | EMA 시간 메모리 |

**EMA 버퍼** (위기 보정 시스템):
- `crisis_bias` [1]: 위기 레벨 중립화 바이어스
- `crisis_temperature` [1]: 위기 시그모이드 스케일링
- `crisis_prev_regime` [1]: 이전 레짐 플래그 (히스테리시스용)
- `crisis_step` [1]: 학습 스텝 카운터
- `sharpe_mean/var` [1]: DSR 신호 정규화 통계
- `cvar_mean/var` [1]: CVaR 신호 정규화 통계
- `crisis_base_mean/var` [1]: T-Cell 베이스 신호 정규화 통계

**Info 구조** (디버그/시각화용):

```python
info = {
    # 프로토타입 믹싱
    'w': [B, M],                    # 최종 혼합 가중치
    'w_rep': [B, M],               # Replicator 출력
    'w_ot': [B, M],                # OT 출력
    'P': [B, m, M],                # 수송 계획
    'fitness': [B, M],             # 프로토타입 적합도
    
    # 위기 감지 (다중 신호)
    'crisis_level': [B, 1],        # 최종 위기 레벨 (guard 적용 후)
    'crisis_level_pre_guard': [B, 1],  # Guard 적용 전
    'crisis_raw': [B, 1],          # 다중 신호 가중 합산
    'crisis_base_component': [B, 1],   # T-Cell 기본 신호 (sigmoid)
    'delta_component': [B, 1],     # DSR 기여도 (sigmoid)
    'cvar_component': [B, 1],      # CVaR 기여도 (sigmoid)
    'crisis_regime': [B, 1],       # 히스테리시스 레짐 {0,1}
    'crisis_types': [B, K=4],      # 위기 타입별 점수
    
    # EMA 보정 상태
    'crisis_bias': [1],            # 바이어스 (target p_star 유도용)
    'crisis_temperature': [1],     # 온도 (스케일 조정용)
    'crisis_prev_regime': [1],     # 이전 레짐
    'crisis_guard_rate': scalar,   # 현재 guard 비율
    
    # 입력 신호 (원본)
    'crisis_base': [B, 1],         # T-Cell 출력 (sigmoid)
    'crisis_base_raw': [B, 1],     # T-Cell 출력 (affine)
    'delta_sharpe': [B, 1],        # DSR bonus
    'cvar': [B, 1],                # CVaR value
    
    # IRT 연산
    'cost_matrix': [B, m, M],      # 면역학적 비용
    'eta': [B, 1],                 # 위기 가열 η(c)
    'alpha_c': [B, 1],             # 동적 혼합 비율
    
    # Dirichlet 정책
    'concentrations': [B, M, action_dim],  # 프로토타입별 concentration
    'mixed_conc': [B, action_dim],         # 혼합 concentration
    'mixed_conc_clamped': [B, action_dim], # Clamp 적용 후
}
```

#### 4. IRT Operator

**역할**: Optimal Transport와 Replicator Dynamics를 동적으로 혼합.

**핵심 수식**:

```
w_t = (1-α_c(c))·Replicator(w_{t-1}, f_t, c) + α_c(c)·Transport(E_t, K, C_t)
```

**세부 구성**:

1. **동적 혼합 비율** α_c(c):
   ```
   α_c(c) = α_max + (α_min - α_max) · (1 - cos(πc)) / 2
   ```
   - c=0 (평시) → α_c=α_max (OT 증가, 구조적 탐색)
   - c=1 (위기) → α_c=α_min (Replicator 증가, 검증된 전략)
   
2. **Sharpe Gradient Feedback** (Phase C):
   ```
   α_c' = α_c·(1 + 0.6·tanh(Δsharpe)) + 0.07·tanh(Δsharpe)
   α_c_final = clamp(α_c', α_min, α_max)
   ```
   - Δsharpe > 0 (성능 상승) → OT 경로 강화 (탐색 지속)
   - Δsharpe < 0 (성능 하락) → Rep 경로 강화 (보수적 전환)

3. **위기 가열** η(c):
   ```
   η(c) = η_0 + η_1·c
   ```
   - 평시: η=0.05 (느린 적응)
   - 위기: η=0.17 (빠른 적응)

#### 5. TCellMinimal (위기 감지)

**역할**: 다중 신호 기반 실시간 위기 감지 및 공자극 임베딩 생성.

**입력**: 시장 특성 [B, 12]
- 시장 통계 (4개): balance, price_mean, price_std, cash_ratio
- 기술 지표 (8개): macd, boll_ub, boll_lb, rsi_30, cci_30, dx_30, sma_30, sma_60

**출력**:
- `z` [B, 4]: 위기 타입별 점수 (Volatility, Liquidity, Correlation, Systemic)
- `danger_embed` [B, D]: 공자극 임베딩 (OT 비용 함수용)
- `crisis_affine` [B, 1]: 가중 합산 (sigmoid 전)
- `crisis_base` [B, 1]: 시그모이드 위기 확률

**아키텍처**:
```python
encoder = Sequential(
    Linear(12, 128),
    LayerNorm(128),
    ReLU(),
    Dropout(0.1),
    Linear(128, 4+D)  # z(4) + danger_embed(D)
)

# 온라인 정규화 (EMA)
z_std = (z - μ) / σ

# 가중 합산
crisis_affine = Σ_k softmax(α)_k · z_std_k
crisis_base = sigmoid(crisis_affine)
```

---

## IRT의 핵심 메커니즘

### 1. Optimal Transport (OT)

**개념**: 현재 상태(에피토프)와 전문가 전략(프로토타입) 간의 **최적 매칭**을 찾는다.

**수학적 배경**:
- Cuturi (2013)의 엔트로피 정규화 최적수송
- Sinkhorn 알고리즘으로 효율적 계산 (O(ε⁻¹log n) 수렴)

**동작 방식**:

```
1. 에피토프 인코딩: E = encoder(state)  # [B, m=6, D=128]
2. 프로토타입 키: K = proto_keys        # [M=8, D]
3. 면역학적 비용 계산:
   C_ij = d_M(E_i, K_j) - γ·<E_i, danger> + λ·tolerance + ρ·checkpoint
   
   - d_M: 학습 가능한 마할라노비스 거리
   - γ·<E_i, danger>: 위기 신호와 정렬된 에피토프 선호
   - λ·tolerance: 자기-내성 (실패 패턴 억제)
   - ρ·checkpoint: 프로토타입 과신 억제
   
4. Sinkhorn 알고리즘:
   P* = argmin_P <P, C> + ε·KL(P || u⊗v^T)
   
   where:
   - u = 1/m · 1_m (균등 소스 분포)
   - v = 1/M · 1_M (균등 타겟 분포)
   - ε: 엔트로피 정규화 (exploration)
   
5. OT 마진 (프로토타입별 수송 질량):
   w_ot[j] = Σ_i P*(i, j)  # [B, M]
```

**직관**:
- 위기 상황 → 위기 신호(`danger`)와 정렬된 에피토프의 비용 ↓
- OT가 자동으로 위기 대응 프로토타입 선택
- 엔트로피 정규화 ε로 exploration-exploitation 균형

### 2. Replicator Dynamics

**개념**: 과거에 성공한 전략을 **선호**하는 시간 메모리 메커니즘 (진화 게임 이론).

**수학적 배경**:
- Hofbauer & Sigmund (1998)의 복제자 동역학
- 균형점은 Nash 균형, 안정점은 ESS (Evolutionarily Stable Strategy)

**동작 방식**:

```
1. 이전 가중치 w_{t-1} 기억 (EMA buffer)

2. 각 프로토타입의 적합도 f_j 계산 (Critic Q-value 기반)
   fitness[j] = min(Q1(s, a_j), Q2(s, a_j))
   where a_j = softmax(decoder_j(proto_key_j))

3. Advantage 계산 (베이스라인 대비):
   baseline = Σ_j w_{t-1,j} · f_j
   A_j = f_j - baseline

4. 위기 가열:
   η(c) = η_0 + η_1 · c
   - 평시 (c≈0): η=0.05 (느린 학습)
   - 위기 (c≈1): η=0.17 (빠른 적응)

5. 자기-내성 페널티:
   proto_self_sim = max_k <K_j, sig_k>  # 실패 패턴 유사도
   r_penalty = 0.5 · proto_self_sim

6. Replicator 방정식 (log-space):
   log_w̃ = log(w_{t-1}) + η · A - r_penalty

7. Temperature softmax (Phase F):
   w_rep = softmax(log_w̃ / τ)
   - τ=1.4 > 1: 분포 평탄화 (균등 고착 해제)
   - τ<1: 집중화 (winner-take-all)
```

**직관**:
- 성공한 프로토타입 (high Q-value) → 가중치 ↑
- 위기 시 → η ↑ (빠른 재할당)
- 자기-내성 → 반복 실수 억제
- Temperature → 다양성 유지

**EMA 시간 메모리**:
```
if training:
    w_prev = β·w_prev + (1-β)·w.mean(dim=0)
    - β=0.70: 메모리 강도
    - 배치 평균으로 안정화
```

### 3. 면역학적 비용 함수

**개념**: 도메인 지식을 비용 함수에 **내장**하여 더 나은 의사결정 유도.

**구성 요소**:

```
C_ij = d_M(E_i, K_j) - γ·co_stim + λ·tolerance + ρ·checkpoint
```

#### 3.1 마할라노비스 거리 (학습 가능)

```
d_M(x, y) = √[(x-y)^T M (x-y)]
where M = L^T L (positive definite)
```

- L은 학습 가능한 하삼각 행렬 (초기값: Identity)
- 데이터 공간의 중요 차원을 자동 발견

#### 3.2 공자극 (Co-stimulation)

```
co_stim_i = <E_i, danger>
```

- 위기 신호(`danger` 임베딩)와 정렬된 에피토프 선호
- 위기 시 위험 신호와 유사한 패턴 우선 선택
- γ=0.85: 공자극 가중치

#### 3.3 내성 (Tolerance)

```
tolerance_penalty_i = ReLU(κ · max_k<E_i, sig_k> - ε_tol)
```

- `sig_k`: 학습 가능한 자기-내성 서명 (실패 패턴)
- 과거 실패와 유사한 에피토프 억제
- κ=1.0: 내성 게인, ε_tol=0.1: 임계값
- λ=2.0: 내성 가중치

#### 3.4 체크포인트 (Checkpoint)

```
checkpoint_penalty_j = ρ · confidence_j
```

- 과신하는 프로토타입 억제
- 과도한 집중 방지 (다양성 유지)
- ρ=0.3: 체크포인트 가중치
- (현재 구현: confidence=0, 향후 확장 가능)

**효과**:
- 단순 거리 기반 매칭보다 **의미 있는 매칭**
- 위기 대응, 실패 회피, 분산 투자 자동 유도
- 학습 가능한 메트릭으로 데이터 적응

---

## 위기 감지 시스템

### 다중 신호 통합

IRT는 세 가지 독립적 신호를 결합하여 위기를 감지한다:

```
crisis_raw = w_r · sigmoid(k_b · crisis_base_z)     # T-Cell 시장 신호
           + w_s · sigmoid(-k_s · sharpe_z)         # DSR gradient (음수: 하락=위기)
           + w_c · sigmoid(k_c · cvar_z)            # CVaR (tail risk)
```

**기본 가중치** (Phase 1 calibration):
- `w_r = 0.55`: T-Cell 시장 신호 (주요 센서)
- `w_s = -0.25`: DSR bonus (성능 하락 감지, 음수로 반전)
- `w_c = 0.20`: CVaR (꼬리 위험)

**각 신호의 역할**:

| 신호 | 측정 대상 | z-score 변환 | Sigmoid 기울기 | 해석 |
|-----|----------|-------------|---------------|-----|
| **T-Cell Base** | 시장 통계 + 기술 지표 | (affine - μ)/σ | k_b=4.0 | 시장 구조적 변화 |
| **DSR Bonus** | Sharpe ratio 변화율 | (Δsharpe - μ)/σ | k_s=6.0 | 성능 변화 추세 |
| **CVaR** | 조건부 VaR (하위 5%) | (cvar - μ)/σ | k_c=6.0 | 극단 손실 위험 |

### EMA 정규화 (Z-Score Normalization)

각 신호는 독립적으로 EMA 통계로 정규화된다:

```python
# 각 신호별 running statistics (momentum=0.95)
signal_z = (signal_raw - μ_signal) / σ_signal

# Sigmoid로 [0,1] 범위 변환
component = sigmoid(k · signal_z)
```

**이점**:
- 신호 간 스케일 통일 (공정한 기여도)
- 비정상성 데이터에 적응 (μ, σ 자동 갱신)
- 극단값에 강건 (outlier 영향 완화)

### 바이어스/온도 보정 (Phase B)

위기 레벨의 전역 분포를 목표(`p_star=0.35`)에 맞추기 위한 EMA 보정:

```
crisis_affine = (crisis_raw - bias) / temperature
crisis_level = sigmoid(crisis_affine)

# 학습 중: EMA 적응 (코사인 감쇠)
if training:
    p_hat = crisis_level.mean()  # 배치 평균
    
    # 바이어스 조정 (목표 점유율로 중립화)
    η_b = cosine_decay(η_b_max=0.02, η_b_min=0.002, steps=30k)
    bias += η_b · (p_hat - p_star)
    
    # 온도 조정 (분산 스케일링)
    η_T = 0.01
    temperature *= (1 + η_T · (p_hat - p_star))
    temperature = clamp(temperature, 0.9, 1.2)
```

**목적**:
- 위기 점유율 안정화 (과도한 위기/평시 편향 방지)
- 신호 드리프트 보정 (비정상성 시계열 적응)
- 학습 초기 빠르게, 후기 느리게 조정 (코사인 감쇠)

### T-Cell Guard (목표 crisis_regime_pct 유도)

위기 레짐 점유율을 목표(`crisis_target=0.5`)로 유도하는 선형 제어:

```
crisis_level_guard = crisis_level_pre_guard 
                   + guard_rate · (target - crisis_level_pre_guard)

# Guard 비율 스케줄 (워밍업)
guard_rate(step) = init + (final - init) · min(step / warmup_steps, 1.0)
```

**기본 설정**:
- `crisis_target = 0.5`: 목표 평형점
- `guard_rate_init = 0.30`: 학습 초기 강한 유도
- `guard_rate_final = 0.05`: 학습 후기/평가 약한 유도
- `warmup_steps = 10000`: 스케줄 워밍업

**효과**:
- 학습 초기: 위기/평시 균형잡힌 경험 확보
- 학습 후기: 자연스러운 위기 감지로 수렴
- 평가 시: guard_rate=final (최소 간섭)

### 히스테리시스 레짐 전환

위기와 평시 간 전환에 히스테리시스를 적용하여 떨림(jitter) 방지:

```
if prev_regime == CRISIS:
    regime = (crisis_level < hysteresis_down) ? NORMAL : CRISIS
else:  # prev_regime == NORMAL
    regime = (crisis_level > hysteresis_up) ? CRISIS : NORMAL
```

**기본 임계값**:
- `hysteresis_up = 0.55`: 평시 → 위기 (상승 진입)
- `hysteresis_down = 0.45`: 위기 → 평시 (하강 진입)

**효과**:
- 레짐 전환 안정화 (경계에서 진동 방지)
- IRT 적응형 메커니즘에 명확한 신호 제공
- 평가 시 위기 구간 식별 용이

---

## 적응형 메커니즘

### 동적 α 혼합 (OT-Replicator Balancing)

위기 레벨에 따라 OT-Replicator 혼합 비율을 자동 조절:

```
α_c(c) = α_max + (α_min - α_max) · (1 - cos(πc)) / 2
```

**기본 범위**:
- `α_min = 0.08`: 위기 시 최소 (Replicator 92%)
- `α_max = 0.45`: 평시 최대 (OT 45%)

**동작**:
- c=0 (완전 평시) → α_c=0.45 (OT 증가, 구조적 탐색)
- c=0.5 (경계) → α_c≈0.27 (균형)
- c=1 (완전 위기) → α_c=0.08 (Replicator 증가, 검증된 전략)

**직관**:
- 평시: 새로운 패턴 탐색 (OT 매칭)
- 위기: 검증된 전략 선호 (Replicator 메모리)

### Sharpe Gradient Feedback (Phase C)

Sharpe ratio 변화율을 α_c에 피드백하여 성능 추세 반영:

```
Δsharpe = DSR_bonus  # 입력 신호 (원본 스케일 ~0.1)
α_c' = α_c · (1 + 0.6·tanh(Δsharpe)) + 0.07·tanh(Δsharpe)
α_c_final = clamp(α_c', α_min, α_max)
```

**동작**:
- Δsharpe > 0 (성능 상승):
  - α_c 증가 → OT 경로 강화 (탐색 지속)
- Δsharpe < 0 (성능 하락):
  - α_c 감소 → Replicator 경로 강화 (보수적 전환)

**이득 증폭** (Phase C calibration):
- 곱셈 항: `0.6·tanh(Δ)` (±60% 변동)
- 가법 항: `0.07·tanh(Δ)` (저α 영역 효과 유지)

**효과**:
- 성능 상승 시 탐색 지속 (좋은 패턴 활용)
- 성능 하락 시 방어 전환 (손실 제한)
- Clamp로 극단 방지 (안정성 확보)

### 위기 가열 (Crisis-Adaptive Learning Rate)

Replicator 학습률을 위기 레벨에 따라 조절:

```
η(c) = η_0 + η_1 · c
```

**기본 설정**:
- `η_0 = 0.05`: 평시 학습률
- `η_1 = 0.12`: 위기 증가량 (Phase E calibration)

**동작**:
- c=0 → η=0.05 (느린 적응, 안정성 우선)
- c=1 → η=0.17 (빠른 적응, 빠른 재할당)

**효과**:
- 평시: 완만한 가중치 변화 (과적합 방지)
- 위기: 신속한 프로토타입 재할당 (위기 대응)

### Replicator Temperature (Phase F)

Replicator softmax에 온도를 적용하여 분포 형태 조절:

```
w_rep = softmax(log_w̃ / τ)
```

**기본 설정**:
- `τ = 1.4` (Phase F calibration)

**동작**:
- τ > 1: 분포 평탄화 (균등 분포에 가까움)
- τ = 1: 표준 softmax
- τ < 1: 분포 집중화 (winner-take-all)

**효과**:
- τ=1.4 → 균등 혼합 고착 해제
- 다양한 프로토타입에 가중치 분산
- 탐색 촉진 (exploration)

---

## 프로토타입 학습

### 프로토타입 구조

IRT는 M개의 전문가 프로토타입을 학습한다:

```python
# 프로토타입 키 (학습 가능)
proto_keys: nn.Parameter [M=8, D=128]
# Xavier 초기화: ~N(0, 1/√D)

# 프로토타입별 Dirichlet 디코더
decoders[j]: Sequential(
    Linear(D, 128),
    ReLU(),
    Dropout(0.1),
    Linear(128, action_dim),
    Tanh()  # [-1, 1] 출력
)
```

**변환**: Tanh → Concentration

```python
# Tanh 출력 [-1,1]을 [0.01, ~10] 범위로 변환
conc_raw = decoder_j(proto_key_j)  # [-1, 1]
conc = clamp(conc_raw * 7.5 + 2.5, min=0.01)  # [0.01, ~10]
```

**매핑**:
- Tanh=-1 → conc=0.01 (극도로 sparse, 특정 자산 집중)
- Tanh=0 → conc=2.5 (약한 분산)
- Tanh=1 → conc=10.0 (강한 분산, 균등 방향)

### 다양성 초기화 (Phase 2.2a)

프로토타입 간 대칭성을 깨기 위한 초기화:

```python
for j, decoder in enumerate(decoders):
    final_linear = decoder[-2]  # 마지막 Linear 레이어
    
    # 분산 초기화 (대칭성 깨기)
    torch.nn.init.normal_(final_linear.bias, mean=0.0, std=0.5)
    
    # 프로토타입별 오프셋 (다양성 유도)
    final_linear.bias += (j - M/2) * 0.2
```

**효과**:
- 프로토타입 j=-1 → bias-0.8 (cash 선호?)
- 프로토타입 j=+1 → bias+0.8 (주식 선호?)
- 학습 초기부터 다른 전략 탐색

### 프로토타입 혼합

IRT 가중치로 프로토타입을 혼합하여 최종 정책 생성:

```python
# 각 프로토타입의 concentration 계산
concentrations = stack([decoders[j](proto_keys[j]) for j in range(M)])
# [B, M, action_dim]

# IRT 가중치로 혼합
mixed_conc = einsum('bm,bma->ba', w, concentrations)
# [B, action_dim]

# 행동 생성
if deterministic:
    action = softmax(mixed_conc / action_temp)  # 결정적
else:
    mixed_conc_clamped = clamp(mixed_conc, dirichlet_min, dirichlet_max)
    action = Dirichlet(mixed_conc_clamped).sample()  # 확률적
```

### Deterministic vs Stochastic

**Deterministic 모드** (평가 시):
```
action = softmax(mixed_conc / τ_action)
```
- Softmax 온도 `τ_action=0.8` (Phase 2)
- 낮은 온도 → 집중된 분포 (exploitation)
- Dirichlet과 **다른** 메커니즘 (logit 기반)

**Stochastic 모드** (학습 시):
```
action ~ Dirichlet(clamp(mixed_conc, min, max))
```
- Concentration clamp: `[0.8, 20.0]` (Phase 3.5)
- α < 1: Sparse (모서리 선호)
- α = 1: Uniform
- α > 1: Peaked (중심 선호)
- Exploration 유지

### Fitness 피드백 루프

프로토타입은 Critic Q-value를 통해 학습된다:

```
1. IRTActorWrapper._compute_fitness(obs):
   - 각 프로토타입 j의 샘플 행동 생성
     a_j = softmax(decoder_j(proto_key_j))
   - Critic Q-value 계산
     fitness[j] = min(Q1(obs, a_j), Q2(obs, a_j))

2. Replicator Dynamics:
   - fitness 높은 프로토타입 → w_rep 증가
   - 학습 시 w_prev EMA 업데이트

3. SAC Actor Loss:
   - action, log_prob = actor.action_log_prob(obs)
   - actor_loss = (α·log_prob - Q(obs, action)).mean()
   - Gradient → decoder 및 proto_keys 업데이트
```

**효과**:
- Q-value 높은 행동 생성 프로토타입 선호
- 다양한 시장 상황에 특화된 전문가 학습
- IRT 혼합으로 상황별 최적 결합

| 메트릭                | SAC Baseline | IRT 목표 (Phase 1.4) | 개선율      |
| --------------------- | ------------ | -------------------- | ----------- |
| **Sharpe Ratio**      | 1.0-1.2      | 1.3-1.5              | **+15-20%** |
| **전체 Max Drawdown** | -30~-35%     | -18~-23%             | **-25-35%** |
| **위기 구간 MDD**     | -40~-45%     | -22~-27%             | **-35-45%** |

**Phase 1.4 개선사항 반영**:

- Replicator 완전 활성화 (0% → 70%)
- TCell 위기 감지 정확도 향상 (시장 통계 + Tech indicators)
- Train-Eval 일관성 확보

### 위기 구간 집중

IRT의 진가는 **위기 구간**에서 발휘된다:

- **2020년 COVID-19**: MDD -40% → -25% (목표)
- **2022년 Fed 금리 인상**: MDD -35% → -22% (목표)
- **정상 구간**: SAC와 유사 (안정성 유지)

### 해석 가능성

IRT는 **블랙박스가 아니다**. 다음 정보를 제공한다:

1. **IRT 분해**: `w = (1-α)·w_rep + α·w_ot`

   - w_rep: 시간 메모리 기여도
   - w_ot: 구조적 매칭 기여도

2. **T-Cell 위기 감지**:

   - 위기 타입별 점수 (변동성, 유동성, 상관관계, 시스템)
   - 위기 레벨 (0~1)

3. **비용 행렬**:

   - 에피토프-프로토타입 간 면역학적 비용
   - 어떤 전략이 왜 선택되었는지 추적

4. **프로토타입 해석**:
   - 각 프로토타입이 선호하는 자산
   - 위기 vs 정상 구간 활성화 패턴

---

## 다른 알고리즘과의 비교

### 호환성 요약

| 알고리즘 | IRT 적용     | Fitness 계산 | Policy 타입   | 권장도     |
| -------- | ------------ | ------------ | ------------- | ---------- |
| **SAC**  | ✅ 최적      | Q(s,a)       | Stochastic    | ⭐⭐⭐⭐⭐ |
| **TD3**  | ✅ 가능      | Q(s,a)       | Deterministic | ⭐⭐⭐⭐   |
| **DDPG** | ✅ 가능      | Q(s,a)       | Deterministic | ⭐⭐⭐     |
| **PPO**  | ⚠️ 수정 필요 | V(s) 기반    | Stochastic    | ⭐⭐       |
| **A2C**  | ⚠️ 수정 필요 | V(s) 기반    | Stochastic    | ⭐⭐       |

### SAC (현재 사용) ⭐⭐⭐⭐⭐

**장점**:

- ✅ **Q-network 기반** → 프로토타입 fitness 계산 용이
- ✅ **Entropy regularization** → IRT exploration과 시너지
- ✅ **Off-policy** → 샘플 효율성 (과거 경험 재사용)
- ✅ **Stochastic policy** → Dirichlet 정책과 완벽 호환
- ✅ **2 Q-networks (ensemble)** → 안정성

**IRT와의 궁합**:

```python
# SAC의 entropy maximization
max E[Q(s,a)] + α_sac·H(π)

# IRT의 exploration
- Sinkhorn entropy (ε)
- Dirichlet concentration (α_k)

# 결과: 이중 exploration → 강건한 학습
```

**사용 예시**:

```python
from stable_baselines3 import SAC
from finrl.agents.irt import IRTPolicy

model = SAC(
    policy=IRTPolicy,
    env=env,
    policy_kwargs={"alpha": 0.3, "eps": 0.10}
)
```

### TD3 ⭐⭐⭐⭐

**장점**:

- ✅ **Q-network 기반** → fitness 계산 가능
- ✅ **Twin Q-networks** → Overestimation 완화
- ✅ **Off-policy** → 샘플 효율성

**차이점**:

- ⚠️ **Deterministic policy** → IRT의 `deterministic=True` 모드 사용
- ❌ **Entropy regularization 없음** → Exploration 약함

**적용 방법**:

```python
from stable_baselines3 import TD3

model = TD3(
    policy=IRTPolicy,  # 동일한 IRT Policy 사용 가능
    env=env,
    policy_kwargs={"alpha": 0.3}
)
# IRT 내부에서 deterministic 모드로 자동 전환
```

### DDPG ⭐⭐⭐

**장점**:

- ✅ **Q-network 기반** → fitness 계산 가능
- ✅ **Off-policy** → 샘플 효율성
- ✅ **단순 구조** → 빠른 학습

**단점**:

- ❌ **Single Q-network** → Overestimation 문제
- ❌ **불안정성** → 학습 발산 가능
- ⚠️ **Deterministic policy**

**권장사항**: TD3 사용 (DDPG 개선 버전)

### PPO ⭐⭐

**문제점**:

- ❌ **V(s) 기반 Critic** → Q(s,a) 없음
- ❌ **On-policy** → 과거 경험 재사용 불가
- ❌ **IRT의 시간 메모리 약화**

**대안** (구조 수정 필요):

```python
# Fitness를 Advantage로 근사
fitness[j] ≈ A(s, a_j) = r + γ·V(s') - V(s)

# 문제:
# 1. Episode 끝까지 기다려야 함 (즉시 계산 불가)
# 2. 분산 ↑ (Monte Carlo 추정)
# 3. On-policy → w_prev 메모리 효과 약화
```

**결론**: IRT와 궁합 나쁨. SAC/TD3 권장.

### A2C ⭐⭐

PPO와 동일한 문제 (V(s) 기반, On-policy).

**추가 단점**:

- Synchronous update → 느린 학습
- PPO의 clipping 없음 → 불안정

---

---

## 하이퍼파라미터 가이드

### IRT 구조 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `emb_dim` | 128 | 64-256 | 임베딩 차원 (에피토프, 프로토타입) |
| `m_tokens` | 6 | 4-8 | 에피토프 토큰 수 (상태 다중 표현) |
| `M_proto` | 8 | 6-12 | 프로토타입 수 (전문가 전략) |
| `market_feature_dim` | 12 | - | T-Cell 입력 차원 (고정) |

### 동적 혼합 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `alpha_min` | 0.08 | 0.05-0.15 | 위기 시 최소 α (Replicator 증가) |
| `alpha_max` | 0.45 | 0.35-0.55 | 평시 최대 α (OT 증가) |
| `ema_beta` | 0.70 | 0.5-0.85 | w_prev EMA 계수 (시간 메모리) |

**α 범위 선택 가이드**:
- `alpha_min` ↓ → 위기 시 Replicator 비중 ↑ (보수적)
- `alpha_max` ↑ → 평시 OT 탐색 ↑ (공격적)
- 기본 설정 (0.08, 0.45): 위기 92% Rep, 평시 45% OT

### 위기 감지 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `w_r` | 0.55 | 0.4-0.7 | T-Cell 시장 신호 가중치 |
| `w_s` | -0.25 | -0.4~-0.1 | DSR bonus 가중치 (음수) |
| `w_c` | 0.20 | 0.1-0.3 | CVaR 가중치 |
| `k_b` | 4.0 | 3.0-6.0 | T-Cell sigmoid 기울기 |
| `k_s` | 6.0 | 4.0-8.0 | DSR sigmoid 기울기 |
| `k_c` | 6.0 | 4.0-8.0 | CVaR sigmoid 기울기 |

**다중 신호 조정**:
- `w_r` ↑ → 시장 구조 민감 (주요 센서)
- `w_s` ↓ (음수 증가) → 성능 하락 민감
- `w_c` ↑ → 꼬리 위험 민감

### 바이어스/온도 보정 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `eta_b` | 0.02 | 0.01-0.05 | 바이어스 초기 학습률 |
| `eta_b_min` | 0.002 | 0.001-0.01 | 바이어스 최소 학습률 (코사인 감쇠 후) |
| `eta_b_decay_steps` | 30000 | 10k-50k | 바이어스 감쇠 스텝 |
| `eta_T` | 0.01 | 0.005-0.02 | 온도 적응 학습률 |
| `p_star` | 0.35 | 0.3-0.5 | 목표 위기 점유율 |
| `temperature_min` | 0.9 | 0.8-1.0 | 온도 하한 |
| `temperature_max` | 1.2 | 1.1-1.5 | 온도 상한 |
| `stat_momentum` | 0.95 | 0.9-0.99 | EMA 정규화 모멘텀 |

### T-Cell Guard 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `crisis_target` | 0.5 | 0.4-0.6 | 목표 crisis_regime_pct |
| `crisis_guard_rate_init` | 0.30 | 0.2-0.5 | 학습 초기 가드 비율 |
| `crisis_guard_rate_final` | 0.05 | 0.0-0.1 | 학습 후기/평가 가드 비율 |
| `crisis_guard_warmup_steps` | 10000 | 5k-20k | 가드 스케줄 워밍업 |
| `hysteresis_up` | 0.55 | 0.5-0.7 | 평시→위기 상승 임계값 |
| `hysteresis_down` | 0.45 | 0.3-0.5 | 위기→평시 하강 임계값 |

### Replicator 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `eta_0` | 0.05 | 0.03-0.08 | 기본 학습률 |
| `eta_1` | 0.12 | 0.05-0.20 | 위기 증가량 (Phase E) |
| `replicator_temp` | 1.4 | 0.9-2.0 | Softmax 온도 (Phase F) |

**Replicator 조정**:
- `eta_0` ↑ → 평시 가중치 변화 ↑
- `eta_1` ↑ → 위기 적응 속도 ↑ (불안정 위험)
- `replicator_temp` ↑ → 분포 평탄화 (다양성 ↑)

### OT (Sinkhorn) 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `eps` | 0.03 | 0.01-0.1 | 엔트로피 정규화 (Phase F2') |
| `max_iters` | 30 | 10-50 | Sinkhorn 최대 반복 |
| `gamma` | 0.85 | 0.5-1.0 | 공자극 가중치 (Phase E) |
| `lambda_tol` | 2.0 | 1.0-3.0 | 내성 가중치 |
| `rho` | 0.3 | 0.1-0.5 | 체크포인트 가중치 |

**OT 조정**:
- `eps` ↑ → 수송 계획 분산 ↑ (exploration)
- `eps` ↓ → 수송 계획 집중 ↑ (exploitation)
- `gamma` ↑ → 위기 신호 민감도 ↑

### Dirichlet 정책 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `dirichlet_min` | 0.8 | 0.1-1.0 | Concentration 최소값 (Phase 3.5) |
| `dirichlet_max` | 20.0 | 5.0-50.0 | Concentration 최대값 (Phase F2') |
| `action_temp` | 0.8 | 0.3-1.0 | 결정적 행동 온도 (Phase 2) |

**Dirichlet 조정**:
- `dirichlet_min` ↓ → 확률적 행동 sparse (exploration)
- `dirichlet_max` ↑ → 확률적 행동 peaked (exploitation)
- `action_temp` ↓ → 결정적 행동 집중 (평가 시)

### 파라미터 Phase 진화

IRT 파라미터는 여러 단계의 실험을 통해 보정되었다:

| Phase | 주요 변경 | 목적 |
|-------|---------|------|
| **Phase 1** | 다중 신호 위기 감지 (w_r, w_s, w_c) | T-Cell + DSR + CVaR 통합 |
| **Phase 2** | action_temp=0.8 추가 | 결정적 행동 민감도 |
| **Phase 2.2a** | dirichlet_min=0.1, ema_beta=0.5 | Exploration 증가 |
| **Phase 3** | ema_beta=0.70 | 전달 감쇠 완화 |
| **Phase 3.5** | dirichlet_min=0.8, alpha_min=0.05 | 균등 흡인 완화, Rep 경로 확보 |
| **Phase B** | 바이어스/온도 보정 도입 | 위기 점유율 안정화 |
| **Phase C** | Sharpe gradient feedback | α_c에 성능 추세 반영 |
| **Phase E** | eta_1=0.12, gamma=0.85 | Replicator 민감도 완화, OT 평활화 |
| **Phase F** | alpha_max=0.45, replicator_temp=1.4 | OT 증가, 분포 평탄화 |
| **Phase F2'** | eps=0.03, dirichlet_max=20.0 | OT 평탄화 완화, 과도 흡인 방지 |

**권장 Ablation Study**:

```bash
# 기본 설정
python scripts/train_irt.py --episodes 200

# alpha 범위 조정
python scripts/train_irt.py --alpha-min 0.05 --alpha-max 0.55

# 위기 신호 가중치 조정
python scripts/train_irt.py --w-r 0.60 --w-s -0.30

# Replicator 가열 조정
python scripts/train_irt.py --eta-1 0.20

# OT 엔트로피 조정
python scripts/train_irt.py --eps 0.05
```

---

## 사용 예시

### 기본 사용법

```python
from stable_baselines3 import SAC
from finrl.agents.irt import IRTPolicy

# IRT Policy 설정
policy_kwargs = {
    # 구조
    "emb_dim": 128,
    "m_tokens": 6,
    "M_proto": 8,
    
    # 동적 혼합
    "alpha_min": 0.08,
    "alpha_max": 0.45,
    "ema_beta": 0.70,
    
    # 위기 감지
    "w_r": 0.55,
    "w_s": -0.25,
    "w_c": 0.20,
    
    # Replicator
    "eta_0": 0.05,
    "eta_1": 0.12,
    "replicator_temp": 1.4,
    
    # OT
    "eps": 0.03,
    "gamma": 0.85,
}

# SAC + IRT
model = SAC(
    policy=IRTPolicy,
    env=env,
    policy_kwargs=policy_kwargs,
    learning_rate=1e-4,
    buffer_size=100000,
    batch_size=64,
    verbose=1
)

# 학습
model.learn(total_timesteps=50000)

# 저장
model.save("logs/irt/irt_model.zip")
```

### CLI 사용법

```bash
# 기본 학습 + 평가
python scripts/train_irt.py --mode both --episodes 200

# 학습만
python scripts/train_irt.py --mode train --episodes 200

# 평가만 (저장된 모델)
python scripts/train_irt.py --mode eval \
  --model logs/irt/20251005_123456/irt_final.zip

# 특정 파라미터로 Ablation Study
python scripts/train_irt.py \
  --episodes 200 \
  --alpha-min 0.05 \
  --alpha-max 0.55 \
  --eta-1 0.15

# SAC Baseline 학습 (비교용)
python scripts/train.py --model sac --mode both --episodes 200
```

### 모델 평가 및 비교

```bash
# 단일 모델 평가
python scripts/evaluate.py \
  --model logs/irt/20251005_123456/irt_final.zip \
  --method direct \
  --save-plot \
  --save-json

# 두 모델 비교 (자동 경로 해석)
python scripts/compare_models.py \
  --model1 logs/sac \
  --model2 logs/irt \
  --output comparison_results

# 특정 타임스탬프 비교
python scripts/compare_models.py \
  --model1 logs/sac/20251005_123456 \
  --model2 logs/irt/20251005_234567 \
  --use-best  # best_model 사용
```

**생성되는 결과물**:

평가 시:
- `evaluation_results.json`: 10개 메트릭 (Sharpe, Calmar, MDD 등)
- `evaluation_plots/`: 14개 시각화 (IRT일 경우)
  - 일반: portfolio_value, returns, drawdown
  - IRT 특화: irt_decomposition, crisis_levels, prototype_weights, tcell_analysis 등

비교 시:
- `comparison_summary.json`: 비교 결과
- `plots/`: 8개 비교 플롯
  - portfolio_value_comparison, drawdown_comparison
  - performance_metrics, risk_metrics
  - rolling_sharpe, crisis_response (IRT)

### IRT 정보 추출 (디버깅/분석)

```python
# 평가 시 IRT 내부 정보 수집
obs = env.reset()
action, _ = model.predict(obs, deterministic=True)

# IRT 정보 가져오기 (IRTPolicy에서 제공)
irt_info = model.policy.get_irt_info()

if irt_info:
    # 프로토타입 혼합
    print(f"프로토타입 가중치: {irt_info['w']}")  # [B, M]
    print(f"Replicator 기여: {irt_info['w_rep']}")
    print(f"OT 기여: {irt_info['w_ot']}")
    
    # 위기 감지
    print(f"위기 레벨: {irt_info['crisis_level']}")  # [B, 1]
    print(f"위기 레짐: {irt_info['crisis_regime']}")  # {0,1}
    print(f"위기 타입별: {irt_info['crisis_types']}")  # [B, 4]
    
    # IRT 연산
    print(f"동적 α: {irt_info['alpha_c']}")  # [B, 1]
    print(f"위기 가열 η: {irt_info['eta']}")  # [B, 1]
    print(f"적합도: {irt_info['fitness']}")  # [B, M]
    
    # 다중 신호
    print(f"T-Cell: {irt_info['crisis_base']}")
    print(f"DSR: {irt_info['delta_sharpe']}")
    print(f"CVaR: {irt_info['cvar']}")
```

### 시각화 자동 생성

IRT 모델 평가 시 14개 시각화가 자동 생성된다:

**일반 플롯** (3개):
- `portfolio_value.png`: 포트폴리오 가치 추이
- `returns_distribution.png`: 수익률 분포 (히스토그램)
- `drawdown.png`: Drawdown 추이

**IRT 특화 플롯** (11개):
- `irt_decomposition.png`: OT vs Replicator 기여도
- `portfolio_weights.png`: 시간별 포트폴리오 가중치
- `crisis_levels.png`: 위기 레벨 추이
- `prototype_weights.png`: 프로토타입 활성화 패턴
- `stock_analysis.png`: 종목별 비중 분석
- `performance_timeline.png`: 성능 시계열
- `benchmark_comparison.png`: 벤치마크 대비 비교
- `risk_dashboard.png`: 위험 지표 대시보드
- `tcell_analysis.png`: T-Cell 위기 타입 분석
- `attribution_analysis.png`: 성능 기여도 분석
- `cost_matrix.png`: OT 비용 행렬 히트맵

**JSON 결과** (2개):
- `evaluation_results.json`: 10개 메트릭
- `evaluation_insights.json`: IRT 특화 인사이트

---

## 참고 문헌

### 이론적 기초

1. **Optimal Transport**
   - Cuturi, M. (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
   - NIPS 2013
   - IRT의 구조적 매칭 메커니즘의 이론적 기반

2. **Replicator Dynamics**
   - Hofbauer, J., & Sigmund, K. (1998). "Evolutionary Games and Population Dynamics"
   - Cambridge University Press
   - IRT의 시간 메모리 메커니즘의 진화 게임 이론적 근거

3. **Information Geometry**
   - Amari, S. (2016). "Information Geometry and Its Applications"
   - Applied Mathematical Sciences
   - Dirichlet 분포 및 정보기하학적 해석

### 강화학습 프레임워크

4. **FinRL**
   - Liu, X. Y., et al. (2024). "FinRL: Financial Reinforcement Learning Framework"
   - NeurIPS Workshop
   - IRT가 통합된 포트폴리오 환경

5. **Stable Baselines3**
   - Raffin, A., et al. (2021). "Stable-Baselines3: Reliable Reinforcement Learning Implementations"
   - JMLR
   - SAC 알고리즘 구현 기반

### 금융 응용

6. **Portfolio Optimization**
   - Markowitz, H. (1952). "Portfolio Selection"
   - Journal of Finance
   - 현대 포트폴리오 이론의 기초

7. **Risk Management**
   - Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of conditional value-at-risk"
   - Journal of Risk
   - CVaR (Conditional Value at Risk) 개념

### 면역학적 영감

8. **Adaptive Immune System**
   - Janeway, C. A., et al. (2001). "Immunobiology"
   - Garland Science
   - T-Cell, B-Cell, 면역학적 메커니즘의 생물학적 배경

---

## 추가 자료

- **프로젝트 문서**: [README.md](../README.md)
- **변경사항 이력**: [CHANGELOG.md](CHANGELOG.md)
- **스크립트 가이드**: [SCRIPTS.md](SCRIPTS.md)
- **FinRL 공식 문서**: [https://finrl.readthedocs.io/](https://finrl.readthedocs.io/)
- **Stable Baselines3 문서**: [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)

---

## 성능 목표 및 기대 효과

### 핵심 목표

IRT는 특히 **위기 구간**에서의 손실 완화를 목표로 한다:

| 메트릭 | SAC Baseline | IRT 목표 | 개선 방향 |
|--------|--------------|----------|-----------|
| **Sharpe Ratio** | 1.0-1.2 | 1.2-1.4 | +10-20% |
| **전체 Max Drawdown** | -30~-35% | -20~-25% | -25-35% |
| **위기 구간 MDD** | -40~-45% | -25~-30% | -30-40% |

**위기 구간 정의**:
- 2020년 COVID-19 충격 (2020-03 ~ 2020-06)
- 2022년 Fed 금리 인상 (2022-01 ~ 2022-12)

### 해석 가능성

IRT는 블랙박스가 아니다. 다음 정보를 제공한다:

1. **IRT 분해**: `w = (1-α_c)·w_rep + α_c·w_ot`
   - Replicator vs OT 기여도 시각화
   - 시간 메모리 vs 구조적 매칭 균형

2. **다중 신호 위기 감지**:
   - T-Cell 시장 신호 (시장 통계 + 기술 지표)
   - DSR bonus (성능 추세)
   - CVaR (꼬리 위험)
   - 각 신호의 기여도 분리

3. **프로토타입 전문성**:
   - 8개 프로토타입의 활성화 패턴
   - 위기 vs 평시 선호도 분석
   - 종목별 특화 전략 식별

4. **비용 행렬 분석**:
   - 에피토프-프로토타입 간 면역학적 비용
   - 어떤 시장 상태가 어떤 전략을 선택했는지 추적
   - 공자극, 내성, 체크포인트 기여도

---

**문의**: GitHub Issues 또는 Discussions 활용

**마지막 업데이트**: 2025-10-13 (현재 코드베이스 반영)
