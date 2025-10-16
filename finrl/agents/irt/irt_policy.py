# finrl/agents/irt/irt_policy.py
# Stable Baselines3 정책과 IRT Actor를 결합하여 위기 적응형 포트폴리오 정책을 제공한다.

"""
Stable Baselines3에서 사용할 수 있는 IRT 커스텀 정책 모듈

이 모듈은 SACPolicy를 확장하여 IRT 기반 Actor를 연결하고,
기존의 Twin Critic 구조를 재사용하면서 면역학적 위기 감지·혼합 로직을 통합한다.

주요 특징:
- SAC의 Critic 두 개를 그대로 활용하여 Q-value를 계산한다.
- Actor 경로는 BCellIRTActor로 대체하여 IRT 연산자를 통한 행동 생성을 수행한다.
- 평가·시각화 시 필요한 IRT 내부 정보를 Actor 래퍼가 유지한다.

간단한 사용 예시는 다음과 같다.

```python
from stable_baselines3 import SAC
from finrl.agents.irt.irt_policy import IRTPolicy

model = SAC(
    policy=IRTPolicy,
    env=env,
    policy_kwargs={
        "emb_dim": 128,
        "m_tokens": 6,
        "M_proto": 8,
        "alpha": 0.3,
    },
)
```
"""

import torch
import torch.nn as nn
import weakref
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces

from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import get_action_dim

from finrl.agents.irt.bcell_actor import BCellIRTActor


class IRTActorWrapper(Actor):
    """
    BCellIRTActor 출력을 Stable Baselines3 Actor 인터페이스에 맞게 노출하는 래퍼

    주요 기능:
    - forward(): 결정적 또는 평균 행동을 반환한다.
    - action_log_prob(): Dirichlet 샘플과 로그 확률을 함께 계산한다.
    - Critic을 활용해 프로토타입별 fitness(적합도)를 추정한다.
    """

    def __init__(
        self,
        irt_actor: BCellIRTActor,
        features_dim: int,
        action_space: spaces.Box,
        policy: Optional['IRTPolicy'] = None
    ):
        """
        Args:
            irt_actor: BCellIRTActor 인스턴스
            features_dim: feature extractor 출력 차원
            action_space: action space
            policy: parent IRTPolicy (Critic 참조용)
        """
        # Actor 부모 클래스 초기화를 건너뛰고 nn.Module만 초기화
        nn.Module.__init__(self)

        self.irt_actor = irt_actor
        self.features_dim = features_dim
        self.action_space = action_space
        self.action_dim = int(action_space.shape[0])

        # Policy를 weakref로 저장 (순환 참조 방지)
        self._policy_ref = weakref.ref(policy) if policy is not None else None

        # Dirichlet 파라미터 저장
        self.dirichlet_min = irt_actor.dirichlet_min
        self.dirichlet_max = irt_actor.dirichlet_max

        # 마지막 forward의 IRT info 저장 (평가/시각화용)
        self._last_irt_info = None

    def _compute_fitness(
        self, obs: torch.Tensor, requires_grad: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Critic을 이용해 프로토타입별 적합도를 산출한다.

        Args:
            obs: [B, features_dim] 형태의 관측 배치

        Returns:
            fitness: [B, M] - 프로토타입별 최소 Q-value, Critic이 없으면 None
        """
        B = obs.size(0)
        M = self.irt_actor.M

        fitness = None

        # WeakRef로부터 policy 인스턴스를 안전하게 꺼낸다.
        policy = self._policy_ref() if self._policy_ref is not None else None

        if policy is not None and hasattr(policy, 'critic'):
            ctx = torch.enable_grad if requires_grad else torch.no_grad
            with ctx():
                # 각 프로토타입에서 행동 후보를 생성한다.
                K = self.irt_actor.proto_keys  # [M, D]
                proto_actions = []

                for j in range(M):
                    conc_j = self.irt_actor.decoders[j](K[j:j+1].expand(B, -1))  # [B, action_dim]
                    conc_j_clamped = torch.clamp(conc_j, min=self.dirichlet_min, max=self.dirichlet_max)
                    a_j = torch.softmax(conc_j_clamped, dim=-1)
                    proto_actions.append(a_j)

                proto_actions = torch.stack(proto_actions, dim=1)  # [B, M, action_dim]

                q_values = []
                for j in range(M):
                    q_vals = policy.critic(obs, proto_actions[:, j])
                    if isinstance(q_vals, tuple):
                        q_min = torch.min(q_vals[0], q_vals[1]).squeeze(-1)
                    else:
                        q_min = q_vals.squeeze(-1) if q_vals.ndim > 1 else q_vals
                    q_values.append(q_min)

                fitness = torch.stack(q_values, dim=1)

        return fitness

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        평균 행동을 계산하여 SAC Actor와 동일한 인터페이스로 반환한다.

        Args:
            obs: [B, features_dim] 형태의 관측 텐서
            deterministic: True일 때는 탐색 없이 평균 행동만 사용한다.

        Returns:
            mean_actions: [B, action_dim] 텐서
        """
        # 입력 dtype을 float32로 맞춰 연산 안정성을 확보한다.
        obs = obs.float()

        # Critic 기반 적합도를 계산한다.
        fitness = self._compute_fitness(obs, requires_grad=False)

        # IRT Actor를 통해 Dirichlet 혼합 행동을 얻는다.
        action, info = self.irt_actor(
            state=obs,
            fitness=fitness,
            deterministic=deterministic,
            retain_grad=False,
        )

        # 평가 및 시각화를 위해 마지막 IRT 결과를 저장한다.
        self._last_irt_info = info

        return action

    def action_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        행동을 샘플링하고 해당 로그 확률을 계산한다.

        Args:
            obs: [B, features_dim] 형태의 관측 텐서

        Returns:
            action: [B, action_dim] - Dirichlet에서 샘플링된 행동
            log_prob: [B, 1] - 해당 행동의 로그 확률
        """
        # 입력 dtype을 float32로 맞춘다.
        obs = obs.float()

        # --- 1단계: Critic으로 프로토타입 적합도를 계산한다. ---
        fitness = self._compute_fitness(obs, requires_grad=False)

        # --- 2단계: 적합도를 반영한 IRT Actor forward를 수행한다. ---
        action, info = self.irt_actor(
            state=obs,
            fitness=fitness,
            deterministic=False,
            retain_grad=False,
        )

        # 해석을 위해 마지막 IRT 결과를 저장한다.
        self._last_irt_info = info

        # --- 3단계: Dirichlet 집중도를 사용해 로그 확률을 계산한다. ---
        mixed_conc_clamped = info['mixed_conc_clamped']

        # Dirichlet 분포의 log_prob를 계산한다.
        dist = torch.distributions.Dirichlet(mixed_conc_clamped)
        log_prob = dist.log_prob(action).unsqueeze(-1)  # [B, 1]

        return action, log_prob

    def get_std(self) -> torch.Tensor:
        """
        SAC의 Actor 인터페이스 호환을 위한 표준편차 근사값을 반환한다.

        Dirichlet 분포의 분산은 α_i(α_0 - α_i) / (α_0^2(α_0 + 1)) 구조를 갖지만,
        여기서는 SAC의 log_std 학습 경로를 사용하지 않으므로 고정값 0.1을 제공한다.
        """
        return torch.ones(self.action_dim) * 0.1


class IRTPolicy(SACPolicy):
    """
    SAC에서 IRT 기반 Actor를 사용하도록 확장한 정책 클래스

    Stable Baselines3의 SACPolicy를 상속하여 Critic 구조는 유지하고,
    Actor 경로를 면역학 기반 BCellIRTActor로 치환한다.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        # IRT-specific parameters
        emb_dim: int = 128,
        m_tokens: int = 6,
        M_proto: int = 8,
        alpha: float = 0.45,  # 평시 기준의 OT 혼합 상한
        alpha_min: float = 0.08,  # 위기 시 Replicator가 최소한 확보해야 하는 비중
        alpha_max: Optional[float] = None,
        ema_beta: float = 0.55,  # IRT Actor EMA 메모리 계수
        market_feature_dim: Optional[int] = None,
        stock_dim: Optional[int] = None,
        tech_indicator_count: Optional[int] = None,
        has_dsr_cvar: bool = False,
        dirichlet_min: float = 0.05,
        dirichlet_max: float = 6.0,
        action_temp: float = 0.50,
        eps: float = 0.03,  # Sinkhorn 엔트로피 계수
        max_iters: int = 30,
        replicator_temp: float = 0.4,
        eta_0: float = 0.05,
        eta_1: float = 0.12,  # 위기 레벨에 따라 추가되는 Replicator 학습률
        alpha_update_rate: float = 0.85,
        alpha_feedback_gain: float = 0.25,
        alpha_feedback_bias: float = 0.0,
        directional_decay_min: float = 0.05,
        alpha_noise_std: float = 0.0,
        gamma: float = 0.90,  # 면역 비용 함수에서 공자극 내적을 차감할 때 사용하는 가중치
        # 위기 감지·보정 기본값
        w_r: float = 0.55,
        w_s: float = -0.25,
        w_c: float = 0.20,
        eta_b: float = 0.02,
        eta_b_min: float = 0.002,
        eta_b_decay_steps: int = 30000,
        eta_T: float = 0.01,
        p_star: float = 0.35,
        temperature_min: float = 0.7,
        temperature_max: float = 1.1,
        stat_momentum: float = 0.92,
        eta_b_warmup_steps: int = 10000,
        eta_b_warmup_value: float = 0.05,
        alpha_crisis_source: str = "pre_guard",
        # T-Cell guard + hysteresis
        crisis_target: float = 0.5,  # 목표 crisis_regime_pct
        crisis_guard_rate_init: float = 0.07,
        crisis_guard_rate_final: float = 0.02,
        crisis_guard_warmup_steps: int = 7500,
        hysteresis_up: float = 0.52,
        hysteresis_down: float = 0.42,
        adaptive_hysteresis: bool = True,
        hysteresis_quantile: float = 0.72,
        hysteresis_min_gap: float = 0.03,
        crisis_history_len: int = 512,
        k_s: float = 6.0,
        k_c: float = 6.0,
        k_b: float = 4.0,
        crisis_guard_rate: Optional[float] = None,
    ):
        """
        Args:
            observation_space: 강화학습 환경의 관측 공간
            action_space: 포트폴리오 가중치가 속한 행동 공간
            lr_schedule: 학습률 스케줄 함수
            net_arch: Critic 신경망 구조 정의
            activation_fn: Critic 활성화 함수
            use_sde: 상태 의존적 탐색 사용 여부
            log_std_init: SAC 기본 Actor용 로그 표준편차 초기값
            use_expln: SAC 기본 Actor의 expln 변환 사용 여부
            clip_mean: SAC 기본 Actor 평균값 클리핑 한계
            features_extractor_class: 특징 추출기 클래스
            features_extractor_kwargs: 특징 추출기 인자
            normalize_images: 이미지 정규화 여부
            optimizer_class: 최적화 알고리즘 클래스
            optimizer_kwargs: 최적화 알고리즘 추가 인자
            n_critics: Critic 개수
            share_features_extractor: Actor·Critic 특징 추출기 공유 여부
            emb_dim: IRT 임베딩 차원
            m_tokens: 에피토프 토큰 개수
            M_proto: 프로토타입 전략 개수
            alpha: OT-Replicator 혼합 비율 기본값 (alpha_max의 초기값으로 활용)
            alpha_min: 위기 상황에서 허용되는 α 최소값
            alpha_max: 평시 α 상한 (None이면 alpha 값을 사용)
            ema_beta: IRT Actor에서 사용하는 EMA 계수
            market_feature_dim: T-Cell 입력으로 사용하는 시장 특성 차원
            stock_dim: 종목 수 (명시하지 않으면 행동 차원에서 유추)
            tech_indicator_count: 기술 지표 개수 (명시하지 않으면 상태 차원에서 유추)
            has_dsr_cvar: 상태에 DSR·CVaR 추가 여부
            dirichlet_min: Dirichlet 집중도 하한 (수치 안정성 확보용)
            dirichlet_max: Dirichlet 집중도 상한
            action_temp: Dirichlet 샘플링 온도
            eps: Sinkhorn 알고리즘의 엔트로피 계수
            max_iters: Sinkhorn 반복 횟수
            replicator_temp: Replicator softmax 온도
            eta_0: Replicator 기본 학습률
            eta_1: 위기 레벨에 비례해 가열되는 추가 학습률
            alpha_update_rate: α 상태 업데이트 속도
            alpha_feedback_gain: Sharpe 변화에 대한 α 증폭 계수
            alpha_feedback_bias: Sharpe 변화에 곱해지는 α 편향 계수
            directional_decay_min: α 방향성 감쇠 최소값
            alpha_noise_std: 학습 시 α에 주입할 가우시안 노이즈 표준편차
            gamma: 면역 비용 함수에서 공자극 내적을 차감할 때 사용하는 가중치
            w_r: 시장 기반 위기 신호 가중치
            w_s: Sharpe 기반 위기 신호 가중치
            w_c: CVaR 기반 위기 신호 가중치
            eta_b: 위기 바이어스를 보정하는 학습률
            eta_b_min: 바이어스 학습률 감쇠 하한
            eta_b_decay_steps: 바이어스 학습률 감쇠에 걸리는 스텝 수
            eta_T: 온도 보정 학습률
            p_star: 목표 위기 점유율
            temperature_min: 위기 온도 하한
            temperature_max: 위기 온도 상한
            stat_momentum: 위기 통계 EMA 모멘텀
            eta_b_warmup_steps: 바이어스 학습률 워밍업 기간
            eta_b_warmup_value: 워밍업 동안 사용할 학습률
            alpha_crisis_source: α 계산에 사용할 위기 신호 시점 (pre_guard 또는 post_guard)
            crisis_target: 위기 레벨이 수렴해야 하는 목표 값
            crisis_guard_rate_init: 초기 위기 가드 강도
            crisis_guard_rate_final: 최종 위기 가드 강도
            crisis_guard_warmup_steps: 위기 가드 강도 전환 기간
            hysteresis_up: 위기 진입 임계값
            hysteresis_down: 위기 해제 임계값
            adaptive_hysteresis: 히스테리시스 임계값의 적응 여부
            hysteresis_quantile: 적응형 히스테리시스에서 사용할 분위수
            hysteresis_min_gap: 적응형 상·하한 사이 최소 간격
            crisis_history_len: 위기 히스토리 버퍼 크기
            k_s: Sharpe 신호 시그모이드 기울기
            k_c: CVaR 신호 시그모이드 기울기
            k_b: 기본 위기 신호 시그모이드 기울기
            crisis_guard_rate: 고정 가드 강도를 강제할 때 사용하는 값
        """
        # IRT 파라미터 저장
        self.emb_dim = emb_dim
        self.m_tokens = m_tokens
        self.M_proto = M_proto
        self.alpha_irt = alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.ema_beta = ema_beta
        self.stock_dim_meta = stock_dim
        self.tech_indicator_count = (
            int(tech_indicator_count) if tech_indicator_count is not None else None
        )
        self.has_dsr_cvar = bool(has_dsr_cvar)
        if market_feature_dim is not None:
            self.market_feature_dim = int(market_feature_dim)
        elif self.tech_indicator_count is not None:
            self.market_feature_dim = 4 + self.tech_indicator_count
        else:
            self.market_feature_dim = 12
        self.dirichlet_min = dirichlet_min
        self.dirichlet_max = dirichlet_max
        self.action_temp = action_temp  # Dirichlet 샘플을 부드럽게 하는 온도 계수
        self.eps = eps
        self.max_iters = max_iters
        self.replicator_temp = replicator_temp
        self.eta_0 = eta_0
        self.eta_1 = eta_1
        self.alpha_update_rate = alpha_update_rate
        self.alpha_feedback_gain = alpha_feedback_gain
        self.alpha_feedback_bias = alpha_feedback_bias
        self.directional_decay_min = directional_decay_min
        self.alpha_noise_std = alpha_noise_std
        self.gamma = gamma
        self.w_r = w_r
        self.w_s = w_s
        self.w_c = w_c
        self.eta_b = eta_b
        self.eta_b_min = eta_b_min
        self.eta_b_decay_steps = eta_b_decay_steps
        self.eta_T = eta_T
        self.p_star = p_star
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.stat_momentum = stat_momentum
        self.eta_b_warmup_steps = eta_b_warmup_steps
        self.eta_b_warmup_value = eta_b_warmup_value
        if alpha_crisis_source not in {"pre_guard", "post_guard"}:
            raise ValueError(
                f"alpha_crisis_source must be 'pre_guard' or 'post_guard', got {alpha_crisis_source}"
            )
        self.alpha_crisis_source = alpha_crisis_source
        self.crisis_target = crisis_target  # T-Cell 가드
        self.crisis_guard_rate_init = crisis_guard_rate_init
        self.crisis_guard_rate_final = crisis_guard_rate_final
        self.crisis_guard_warmup_steps = crisis_guard_warmup_steps
        self.hysteresis_up = hysteresis_up
        self.hysteresis_down = hysteresis_down
        self.adaptive_hysteresis = adaptive_hysteresis
        self.hysteresis_quantile = hysteresis_quantile
        self.hysteresis_min_gap = hysteresis_min_gap
        self.crisis_history_len = crisis_history_len
        self.k_s = k_s
        self.k_c = k_c
        self.k_b = k_b
        self.crisis_guard_rate = crisis_guard_rate  # 지정된 경우 과거 설정과 동일한 고정 가드 비율을 사용한다.

        # SACPolicy 초기화
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> IRTActorWrapper:
        """
        IRT Actor를 생성한 뒤 Stable Baselines3 형식으로 감싼 객체를 반환한다.

        SACPolicy의 make_actor를 재정의하여 BCellIRTActor를 구성하고,
        Critic 정보에 접근할 수 있도록 IRTActorWrapper와 결합한다.
        """
        # 특징 추출기가 지정되지 않은 경우 기본 구현을 사용한다.
        if features_extractor is None:
            features_extractor = self.make_features_extractor()

        features_dim = features_extractor.features_dim
        action_dim = get_action_dim(self.action_space)

        # 포트폴리오 행동 생성을 담당할 BCellIRTActor를 초기화한다.
        bcell_actor = BCellIRTActor(
            state_dim=features_dim,
            action_dim=action_dim,
            emb_dim=self.emb_dim,
            m_tokens=self.m_tokens,
            M_proto=self.M_proto,
            alpha=self.alpha_irt,
            alpha_min=self.alpha_min,
            alpha_max=self.alpha_max,
            ema_beta=self.ema_beta,
            market_feature_dim=self.market_feature_dim,
            stock_dim=self.stock_dim_meta or action_dim,
            tech_indicator_count=self.tech_indicator_count,
            has_dsr_cvar=self.has_dsr_cvar,
            dirichlet_min=self.dirichlet_min,
            dirichlet_max=self.dirichlet_max,
            action_temp=self.action_temp,  # Dirichlet 샘플 온도
            eps=self.eps,
            max_iters=self.max_iters,
            replicator_temp=self.replicator_temp,
            eta_0=self.eta_0,
            eta_1=self.eta_1,
            alpha_update_rate=self.alpha_update_rate,
            alpha_feedback_gain=self.alpha_feedback_gain,
            alpha_feedback_bias=self.alpha_feedback_bias,
            directional_decay_min=self.directional_decay_min,
            alpha_noise_std=self.alpha_noise_std,
            gamma=self.gamma,
            w_r=self.w_r,
            w_s=self.w_s,
            w_c=self.w_c,
            eta_b=self.eta_b,
            eta_b_min=self.eta_b_min,
            eta_b_decay_steps=self.eta_b_decay_steps,
            eta_T=self.eta_T,
            p_star=self.p_star,
            temperature_min=self.temperature_min,
            temperature_max=self.temperature_max,
            stat_momentum=self.stat_momentum,
            eta_b_warmup_steps=self.eta_b_warmup_steps,
            eta_b_warmup_value=self.eta_b_warmup_value,
            alpha_crisis_source=self.alpha_crisis_source,
            crisis_target=self.crisis_target,  # T-Cell 가드
            crisis_guard_rate_init=self.crisis_guard_rate_init,
            crisis_guard_rate_final=self.crisis_guard_rate_final,
            crisis_guard_warmup_steps=self.crisis_guard_warmup_steps,
            hysteresis_up=self.hysteresis_up,
            hysteresis_down=self.hysteresis_down,
            adaptive_hysteresis=self.adaptive_hysteresis,
            hysteresis_quantile=self.hysteresis_quantile,
            hysteresis_min_gap=self.hysteresis_min_gap,
            crisis_history_len=self.crisis_history_len,
            k_s=self.k_s,
            k_c=self.k_c,
            k_b=self.k_b,
            crisis_guard_rate=self.crisis_guard_rate  # 지정된 경우 고정된 가드 강도를 사용한다.
        )

        # Critic 참조를 유지하기 위해 현재 정책을 전달한 래퍼를 생성한다.
        actor = IRTActorWrapper(
            irt_actor=bcell_actor,
            features_dim=features_dim,
            action_space=self.action_space,
            policy=self
        )

        return actor

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """SB3 체크포인트 저장용 파라미터"""
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                emb_dim=self.emb_dim,
                m_tokens=self.m_tokens,
                M_proto=self.M_proto,
                alpha=self.alpha_irt,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max,
                ema_beta=self.ema_beta,
                market_feature_dim=self.market_feature_dim,
                stock_dim=self.stock_dim_meta,
                tech_indicator_count=self.tech_indicator_count,
                has_dsr_cvar=self.has_dsr_cvar,
                dirichlet_min=self.dirichlet_min,
                dirichlet_max=self.dirichlet_max,
                action_temp=self.action_temp,
                eps=self.eps,
                max_iters=self.max_iters,
                replicator_temp=self.replicator_temp,
                eta_0=self.eta_0,
                eta_1=self.eta_1,
                alpha_update_rate=self.alpha_update_rate,
                alpha_feedback_gain=self.alpha_feedback_gain,
                alpha_feedback_bias=self.alpha_feedback_bias,
                directional_decay_min=self.directional_decay_min,
                gamma=self.gamma,
                w_r=self.w_r,
                w_s=self.w_s,
                w_c=self.w_c,
                eta_b=self.eta_b,
                eta_b_min=self.eta_b_min,
                eta_b_decay_steps=self.eta_b_decay_steps,
                eta_T=self.eta_T,
                p_star=self.p_star,
                temperature_min=self.temperature_min,
                temperature_max=self.temperature_max,
                stat_momentum=self.stat_momentum,
                eta_b_warmup_steps=self.eta_b_warmup_steps,
                eta_b_warmup_value=self.eta_b_warmup_value,
                alpha_crisis_source=self.alpha_crisis_source,
                crisis_target=self.crisis_target,
                crisis_guard_rate_init=self.crisis_guard_rate_init,
                crisis_guard_rate_final=self.crisis_guard_rate_final,
                crisis_guard_warmup_steps=self.crisis_guard_warmup_steps,
                hysteresis_up=self.hysteresis_up,
                hysteresis_down=self.hysteresis_down,
                adaptive_hysteresis=self.adaptive_hysteresis,
                hysteresis_quantile=self.hysteresis_quantile,
                hysteresis_min_gap=self.hysteresis_min_gap,
                crisis_history_len=self.crisis_history_len,
                k_s=self.k_s,
                k_c=self.k_c,
                k_b=self.k_b,
                crisis_guard_rate=self.crisis_guard_rate,
            )
        )
        return data

    def get_irt_info(self) -> Optional[Dict]:
        """
        최근 forward 호출에서 수집된 IRT 해석 정보를 반환한다.

        Returns:
            dict: 아래 항목을 포함하는 사전
                - w: [B, M] - 최종 프로토타입 혼합 가중치
                - w_rep: [B, M] - Replicator 경로 결과
                - w_ot: [B, M] - Optimal Transport 경로 질량
                - crisis_level: [B, 1] - 가드 적용 이후의 위기 레벨
                - crisis_level_pre_guard: [B, 1] - 가드 적용 이전 위기 확률
                - crisis_raw: [B, 1] - 위기 신호 가중합
                - crisis_bias / crisis_temperature: [1] - EMA 기반 보정 상태
                - crisis_prev_regime: [1] - 직전 레짐 플래그
                - crisis_base_component / delta_component / cvar_component: [B, 1] - 개별 신호 기여도
                - crisis_regime: [B, 1] - 히스테리시스 기반 레짐 판단
                - crisis_guard_rate: float - 현재 가드 강도
                - crisis_types: [B, K] - 위기 타입별 점수
                - cost_matrix: [B, m, M] - 면역 비용 행렬
                - P: [B, m, M] - 수송 계획
                - fitness: [B, M] - 프로토타입 적합도
                - eta: [B, 1] - 위기 적응형 Replicator 학습률
                - alpha_c: [B, 1] - 동적 OT-Replicator 혼합 비율
                - crisis_base / crisis_base_raw: [B, 1] - T-Cell 기준 신호 (sigmoid/logit)
                - delta_sharpe / cvar: [B, 1] - 입력 Sharpe·CVaR 변화량
            None: 정책이 아직 행동을 생성하지 않았을 때
        """
        if hasattr(self, 'actor') and hasattr(self.actor, '_last_irt_info'):
            return self.actor._last_irt_info
        return None


class IRTActorCriticPolicy(IRTPolicy):
    """
    IRT Actor와 SB3 기본 Critic 구성을 그대로 유지하기 위한 호환성 클래스

    SAC는 Actor와 Critic을 분리해 관리하므로, 대부분의 사용자는 IRTPolicy만 지정하면 된다.
    다만 SB3의 일부 API와의 호환을 위해 명시적 ActorCritic 정책을 제공한다.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# 과거 버전과의 호환을 위한 별칭
IRT_Policy = IRTPolicy
