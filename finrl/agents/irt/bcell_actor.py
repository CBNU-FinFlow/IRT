# finrl/agents/irt/bcell_actor.py
# 상태 인코딩, IRT 연산, Dirichlet 디코딩을 결합한 B-Cell Actor 구현을 제공한다.

"""
IRT(Immune Replicator Transport) 기반 B-Cell Actor

구성 요소:
1. 상태 임베딩 → 다중 에피토프 토큰 인코딩
2. T-Cell을 통한 위기 감지 및 공자극 임베딩 생성
3. IRT 연산자를 이용한 Replicator·Optimal Transport 혼합
4. Dirichlet 디코더와 EMA 메모리를 통해 포트폴리오 가중치 산출

해당 Actor는 `IRTPolicy`가 Stable Baselines3와 연동될 때 사용되는 핵심 모듈이다.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Any

from finrl.agents.irt.irt_operator import IRT
from finrl.agents.irt.t_cell import TCellMinimal


class BCellIRTActor(nn.Module):
    """상태 인코딩, 위기 감지, IRT 연산, Dirichlet 디코더를 결합한 포트폴리오 Actor"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        emb_dim: int = 128,
        m_tokens: int = 6,
        M_proto: int = 8,
        alpha: float = 0.3,
        alpha_min: float = 0.06,
        alpha_max: Optional[float] = None,
        ema_beta: float = 0.55,  # 혼합 가중치가 적합도 변화에 빠르게 반응하도록 설정된 EMA 계수
        market_feature_dim: Optional[int] = None,
        stock_dim: Optional[int] = None,
        tech_indicator_count: Optional[int] = None,
        has_dsr_cvar: bool = False,
        dirichlet_min: float = 0.05,
        dirichlet_max: float = 6.0,
        action_temp: float = 0.50,
        # 다중 신호 기반 위기 감지 가중치
        w_r: float = 0.55,
        w_s: float = -0.25,
        w_c: float = 0.20,
        # 위기 바이어스 EMA 보정 파라미터
        eta_b: float = 0.02,
        eta_b_min: float = 0.002,
        eta_b_decay_steps: int = 30000,
        eta_T: float = 0.01,
        eta_b_warmup_steps: int = 10000,
        eta_b_warmup_value: float = 0.05,
        # T-Cell 가드
        crisis_target: float = 0.50,  # 목표 crisis_regime_pct
        crisis_guard_rate_init: float = 0.07,
        crisis_guard_rate_final: float = 0.02,
        crisis_guard_warmup_steps: int = 7500,
        # 히스테리시스 임계치 기본값
        hysteresis_up: float = 0.52,
        hysteresis_down: float = 0.42,
        adaptive_hysteresis: bool = True,
        # 적응형 히스테리시스를 제어하는 분위수
        hysteresis_quantile: float = 0.72,
        hysteresis_min_gap: float = 0.03,
        crisis_history_len: int = 512,
        crisis_guard_rate: Optional[float] = None,
        k_s: float = 6.0,
        k_c: float = 6.0,
        k_b: float = 4.0,
        p_star: float = 0.35,
        temperature_min: float = 0.7,
        temperature_max: float = 1.1,
        stat_momentum: float = 0.92,
        alpha_crisis_source: str = "pre_guard",
        **irt_kwargs
    ):
        """
        Args:
            state_dim: 관측 상태 벡터 차원
            action_dim: 포트폴리오 가중치 개수
            emb_dim: 에피토프 임베딩 차원
            m_tokens: 에피토프 토큰 개수
            M_proto: 학습되는 프로토타입 전략 개수
            alpha: OT-Replicator 혼합 비율의 기본값 (alpha_max 초기값으로 사용)
            alpha_min: 위기 상황에서 Replicator가 보장해야 하는 최소 비중
            alpha_max: 평시 허용되는 α 상한 (None이면 alpha를 사용)
            ema_beta: 혼합 가중치 EMA 메모리 계수
            market_feature_dim: T-Cell 입력으로 사용하는 시장 특성 차원
            stock_dim: 종목 수 (지정하지 않으면 action_dim으로 설정)
            tech_indicator_count: 기술 지표 개수 (자동 추정 가능)
            has_dsr_cvar: 상태에 DSR·CVaR 특성을 포함하는지 여부
            dirichlet_min: Dirichlet 집중도 하한
            dirichlet_max: Dirichlet 집중도 상한
            action_temp: Dirichlet 샘플 온도
            w_r: 시장 기반 위기 신호 가중치
            w_s: Sharpe 기반 위기 신호 가중치
            w_c: CVaR 기반 위기 신호 가중치
            eta_b: 위기 바이어스 학습률 최대값
            eta_b_min: 위기 바이어스 학습률 최소값
            eta_b_decay_steps: 학습률이 최대값에서 최소값으로 감쇠되는 스텝 수
            eta_T: 위기 온도 보정 학습률
            eta_b_warmup_steps: 위기 바이어스 학습률 워밍업 기간
            eta_b_warmup_value: 워밍업 기간 동안 사용할 학습률
            crisis_target: 목표 위기 비중
            crisis_guard_rate_init: 초기 위기 가드 강도
            crisis_guard_rate_final: 최종 위기 가드 강도
            crisis_guard_warmup_steps: 위기 가드 강도 전환 기간
            hysteresis_up: 위기 진입 히스테리시스 상한
            hysteresis_down: 위기 해제 히스테리시스 하한
            adaptive_hysteresis: 히스테리시스 임계값을 적응적으로 조정할지 여부
            hysteresis_quantile: 적응형 히스테리시스에서 사용할 분위수
            hysteresis_min_gap: 적응형 상·하한 사이 최소 간격
            crisis_history_len: 위기 히스토리 버퍼 길이
            crisis_guard_rate: 고정된 위기 가드 강도를 강제할 때 사용
            k_s: Sharpe 신호 변환 기울기
            k_c: CVaR 신호 변환 기울기
            k_b: 기본 위기 신호 변환 기울기
            p_star: 목표 위기 점유율
            temperature_min: 위기 온도 하한
            temperature_max: 위기 온도 상한
            stat_momentum: 위기 통계 EMA 모멘텀
            alpha_crisis_source: α 계산에 사용할 위기 신호 시점
            **irt_kwargs: IRT 연산자에 전달할 추가 파라미터 (예: eps, eta_0 등)
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.emb_dim = emb_dim
        self.m = m_tokens
        self.M = M_proto
        self.ema_beta = ema_beta
        self.dirichlet_min = dirichlet_min
        self.dirichlet_max = dirichlet_max
        self.action_temp = action_temp  # Dirichlet 샘플 온도
        self.has_dsr_cvar = bool(has_dsr_cvar)
        self.stock_dim = int(stock_dim) if stock_dim is not None else action_dim
        if self.stock_dim <= 0:
            raise ValueError(f"stock_dim must be positive, got {self.stock_dim}")
        extra_dims = 2 if self.has_dsr_cvar else 0
        inferred_indicator_count = tech_indicator_count
        if inferred_indicator_count is None:
            remainder = max(state_dim - (1 + 2 * self.stock_dim) - extra_dims, 0)
            inferred_indicator_count = remainder // self.stock_dim if self.stock_dim > 0 else 0
        self.tech_indicator_count = int(max(inferred_indicator_count or 0, 0))
        computed_market_dim = 4 + self.tech_indicator_count
        if market_feature_dim is not None and market_feature_dim >= computed_market_dim:
            self.market_feature_dim = int(market_feature_dim)
        else:
            self.market_feature_dim = computed_market_dim
        self.market_feature_dim = max(self.market_feature_dim, 4)

        # 위기 신호 결합에 사용되는 가중치
        self.w_r = w_r
        self.w_s = w_s
        self.w_c = w_c

        # 위기 신호를 시그모이드로 변환할 때 사용하는 기울기
        self.k_s = k_s
        self.k_c = k_c
        self.k_b = k_b

        # 위기 바이어스와 온도 보정을 위한 하이퍼파라미터
        self.eta_b_max = eta_b
        self.eta_b_min = eta_b_min
        self.eta_b_decay_steps = eta_b_decay_steps
        self.eta_T = eta_T
        self.p_star = p_star
        self.eta_b_warmup_steps = max(int(eta_b_warmup_steps), 0)
        self.eta_b_warmup_value = float(eta_b_warmup_value)
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.stat_momentum = stat_momentum
        self.stats_eps = 1e-6
        self._orientation_checked = False
        if alpha_crisis_source not in {"pre_guard", "post_guard"}:
            raise ValueError(
                f"alpha_crisis_source must be 'pre_guard' or 'post_guard', got {alpha_crisis_source}"
            )
        self.alpha_crisis_source = alpha_crisis_source

        self.register_buffer("crisis_bias", torch.zeros(1))
        self.register_buffer("crisis_temperature", torch.ones(1))
        self.register_buffer("crisis_prev_regime", torch.zeros(1))
        self.register_buffer("crisis_step", torch.zeros(1))
        self.register_buffer("sharpe_mean", torch.zeros(1))
        self.register_buffer("sharpe_var", torch.ones(1))
        self.register_buffer("cvar_mean", torch.zeros(1))
        self.register_buffer("cvar_var", torch.ones(1))
        self.register_buffer("crisis_base_mean", torch.zeros(1))
        self.register_buffer("crisis_base_var", torch.ones(1))
        self.register_buffer("bias_warm_started", torch.zeros(1))
        self.register_buffer("crisis_robust_loc", torch.zeros(1))
        self.register_buffer("crisis_robust_scale", torch.ones(1))
        self._crisis_initialized = False
        self._verify_signal_orientation()

        # T-Cell 가드
        self.crisis_target = crisis_target
        self.crisis_guard_rate_init = crisis_guard_rate_init
        self.crisis_guard_rate_final = crisis_guard_rate_final
        self.crisis_guard_warmup_steps = crisis_guard_warmup_steps
        self.hysteresis_up = hysteresis_up
        self.hysteresis_down = hysteresis_down
        self.adaptive_hysteresis = adaptive_hysteresis
        self.hysteresis_quantile = float(max(min(hysteresis_quantile, 0.99), 0.5))
        self.hysteresis_min_gap = float(max(hysteresis_min_gap, 0.01))
        history_len = max(int(crisis_history_len), 64)
        self.register_buffer("crisis_history", torch.zeros(history_len))
        self.register_buffer("crisis_hist_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("crisis_hist_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("hysteresis_up_state", torch.tensor(float(hysteresis_up)))
        self.register_buffer(
            "hysteresis_down_state", torch.tensor(float(hysteresis_down))
        )
        if crisis_guard_rate is not None:
            self.crisis_guard_rate_init = float(crisis_guard_rate)
            self.crisis_guard_rate_final = float(crisis_guard_rate)
            self.crisis_guard_warmup_steps = 0

        # ===== 에피토프 인코더 =====
        self.epitope_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, m_tokens * emb_dim),
        )

        # ===== 프로토타입 키 (학습 가능) =====
        # Xavier 초기화
        self.proto_keys = nn.Parameter(torch.randn(M_proto, emb_dim) / (emb_dim**0.5))

        # ===== 프로토타입별 Dirichlet 디코더 =====
        # 각 프로토타입이 독립적인 행동 전문가처럼 작동하도록 구성한다.
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(emb_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, action_dim),
                    nn.Tanh(),  # [-1, 1] 범위로 출력해 프로토타입 억제·강화를 모두 허용한다.
                )
                for _ in range(M_proto)
            ]
        )

        # 초기 편향을 다양화하여 프로토타입이 서로 다른 전략을 학습하도록 돕는다.
        for decoder_idx, decoder in enumerate(self.decoders):
            # decoder[-2]는 Tanh 이전의 마지막 선형 계층이다.
            final_linear = decoder[-2]
            # 분산이 있는 초기화를 사용해 대칭 해를 피한다.
            torch.nn.init.normal_(final_linear.bias, mean=0.0, std=0.5)
            # 프로토타입마다 서로 다른 초기 편향을 부여한다.
            final_linear.bias.data += (decoder_idx - self.M / 2) * 0.2

        # ===== IRT 연산자 =====
        self.irt = IRT(
            emb_dim=emb_dim,
            m_tokens=m_tokens,
            M_proto=M_proto,
            alpha=alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            **irt_kwargs
        )

        # ===== T-Cell 통합 =====
        self.t_cell = TCellMinimal(in_dim=self.market_feature_dim, emb_dim=emb_dim)

        # ===== 이전 가중치 (EMA) =====
        self.register_buffer("w_prev", torch.full((1, M_proto), 1.0 / M_proto))

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """
        초기 버전 체크포인트와의 호환성을 유지하기 위한 로더

        누락된 버퍼는 기본값으로 채운 뒤 상위 클래스를 호출한다.
        """
        compatibility_buffers = [
            "crisis_bias",
            "crisis_temperature",
            "crisis_prev_regime",
            "crisis_step",
            "sharpe_mean",
            "sharpe_var",
            "cvar_mean",
            "cvar_var",
            "crisis_base_mean",
            "crisis_base_var",
            "bias_warm_started",
        ]

        for name in compatibility_buffers:
            full_key = prefix + name
            if full_key not in state_dict:
                state_dict[full_key] = getattr(self, name).detach().clone()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _current_eta_b(self) -> float:
        step = float(self.crisis_step.item())
        if step < self.eta_b_warmup_steps:
            return self.eta_b_warmup_value
        adjusted_step = max(step - self.eta_b_warmup_steps, 0.0)
        if self.eta_b_decay_steps <= 0:
            return self.eta_b_max
        ratio = min(max(adjusted_step / float(self.eta_b_decay_steps), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.eta_b_min + (self.eta_b_max - self.eta_b_min) * cosine

    def _current_guard_rate(self) -> float:
        if not self.training:
            return self.crisis_guard_rate_final
        if self.crisis_guard_warmup_steps <= 0:
            return self.crisis_guard_rate_final
        step = float(self.crisis_step.item())
        progress = min(max(step / float(self.crisis_guard_warmup_steps), 0.0), 1.0)
        return (
            self.crisis_guard_rate_init
            + (self.crisis_guard_rate_final - self.crisis_guard_rate_init) * progress
        )

    def _update_hysteresis_thresholds(self, crisis_samples: torch.Tensor) -> None:
        if not self.adaptive_hysteresis or not self.training:
            return
        value = float(crisis_samples.detach().mean().clamp(0.0, 1.0).item())
        ptr = int(self.crisis_hist_ptr.item())
        self.crisis_history[ptr] = value
        ptr = (ptr + 1) % self.crisis_history.numel()
        self.crisis_hist_ptr.fill_(ptr)
        count = int(self.crisis_hist_count.item())
        count = min(count + 1, self.crisis_history.numel())
        self.crisis_hist_count.fill_(count)
        if count < 32:
            return
        history = self.crisis_history[:count].to(crisis_samples.device)
        quantile = torch.quantile(history, self.hysteresis_quantile).item()
        up = max(min(quantile, 1.0), 0.0)
        down = max(up - self.hysteresis_min_gap, 0.0)
        self.hysteresis_up_state.fill_(up)
        self.hysteresis_down_state.fill_(down)

    def _current_hysteresis_thresholds(self) -> Tuple[float, float]:
        if self.adaptive_hysteresis:
            up = float(self.hysteresis_up_state.item())
            down = float(self.hysteresis_down_state.item())
        else:
            up = self.hysteresis_up
            down = self.hysteresis_down
        return up, down

    def _verify_signal_orientation(self) -> None:
        if self._orientation_checked:
            return

        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        sharpe_good = sigmoid(-self.k_s * 1.0)
        sharpe_bad = sigmoid(-self.k_s * -1.0)
        cvar_good = sigmoid(self.k_c * 1.0)
        cvar_bad = sigmoid(self.k_c * -1.0)
        base_good = sigmoid(self.k_b * 1.0)
        base_bad = sigmoid(self.k_b * -1.0)

        if not (sharpe_good < sharpe_bad):
            raise ValueError(
                "Sharpe signal transform orientation invalid (expected Sharpe↑ → crisis↓)."
            )
        if not (cvar_good > cvar_bad):
            raise ValueError(
                "CVaR signal transform orientation invalid (expected CVaR↑ → crisis↑)."
            )
        if not (base_good > base_bad):
            raise ValueError(
                "Base crisis transform orientation invalid (expected base↑ → crisis↑)."
            )
        self._orientation_checked = True

    def forward(
        self,
        state: torch.Tensor,
        fitness: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        retain_grad: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            state: [B, S]
            fitness: 프로토타입 적합도 [B, M] (optional, 없으면 균등)
            deterministic: 결정적 행동 (평가 시)
            retain_grad: True면 info에 gradient가 유지된 텐서를 포함

        Returns:
            action: [B, A] - 포트폴리오 가중치
            info: 해석 정보 (w, P, crisis 등)
        """
        B = state.size(0)

        if not self._crisis_initialized:
            self._crisis_initialized = True

        # --- 1단계: T-Cell 위기 감지 ---
        # FinRL 상태 벡터는 [balance(1), prices(N), shares(N), tech_indicators(K*N), ... 추가 항목] 구조를 따른다.

        balance = state[:, 0:1]  # [B, 1]
        stock_dim = self.stock_dim
        price_start = 1
        price_end = price_start + stock_dim
        prices = state[:, price_start:price_end]  # [B, stock_dim]

        shares_start = price_end
        shares_end = shares_start + stock_dim
        shares = state[:, shares_start:shares_end]  # [B, stock_dim]

        # 시장 통계량
        price_mean = prices.mean(dim=1, keepdim=True)  # [B, 1]
        price_std = prices.std(dim=1, keepdim=True) + 1e-8  # [B, 1]
        total_value = balance + (prices * shares).sum(dim=1, keepdim=True)  # [B, 1]
        cash_ratio = balance / (total_value + 1e-8)  # [B, 1]

        # 기술 지표는 각 지표별 첫 번째 종목 값을 사용한다.
        tech_start = shares_end
        dsr_offset = 2 if self.has_dsr_cvar else 0
        available = max(state.size(1) - dsr_offset - tech_start, 0)
        indicator_blocks = min(self.tech_indicator_count, available // stock_dim)
        tech_features = None
        if indicator_blocks > 0:
            tech_total = indicator_blocks * stock_dim
            tech_slice = state[:, tech_start : tech_start + tech_total]
            tech_features_full = tech_slice.view(B, indicator_blocks, stock_dim)
            tech_features = tech_features_full[:, :, 0]  # [B, indicator_blocks]
        else:
            tech_features = torch.zeros(
                (B, 0), device=state.device, dtype=state.dtype
            )

        market_components = [balance, price_mean, price_std, cash_ratio]
        if tech_features.shape[1] > 0:
            market_components.append(tech_features)
        market_features = torch.cat(market_components, dim=1)
        if market_features.shape[1] < self.market_feature_dim:
            pad_cols = self.market_feature_dim - market_features.shape[1]
            market_features = torch.cat(
                [market_features, torch.zeros(B, pad_cols, device=state.device, dtype=state.dtype)],
                dim=1,
            )
        elif market_features.shape[1] > self.market_feature_dim:
            market_features = market_features[:, : self.market_feature_dim]

        z, danger_embed, crisis_affine_raw, crisis_base_sigmoid = self.t_cell(
            market_features, update_stats=self.training
        )

        def _update_running_stats(
            sample: torch.Tensor, mean_buf: torch.Tensor, var_buf: torch.Tensor
        ) -> None:
            if not self.training:
                return
            if sample.ndim == 1:
                sample_view = sample.view(-1, 1)
            else:
                sample_view = sample
            with torch.no_grad():
                batch_mean = sample_view.detach().mean(dim=0)
                if sample_view.size(0) > 1:
                    batch_var = sample_view.detach().var(dim=0, unbiased=False)
                else:
                    batch_var = torch.zeros_like(batch_mean)
                mean_buf.mul_(self.stat_momentum).add_(
                    batch_mean * (1 - self.stat_momentum)
                )
                var_buf.mul_(self.stat_momentum).add_(
                    batch_var * (1 - self.stat_momentum)
                )
                var_buf.clamp_(min=self.stats_eps)

        # T-Cell 기반 기본 위기 신호를 계산한다.
        _update_running_stats(
            crisis_affine_raw, self.crisis_base_mean, self.crisis_base_var
        )
        base_mean = self.crisis_base_mean.to(state.device)
        base_std = torch.sqrt(self.crisis_base_var.to(state.device) + self.stats_eps)
        base_z = (crisis_affine_raw - base_mean) / base_std
        crisis_base_component = torch.sigmoid(self.k_b * base_z)

        # DSR/CVaR 신호를 추출해 위기 레벨 계산에 반영한다.
        # state 마지막 2개 차원: [dsr_bonus, cvar_value]
        # state_dim은 reward_type='dsr_cvar'일 때 +2 되어 있음
        # 따라서 항상 마지막 2개 차원을 시도하되, 존재하지 않으면 0으로 처리
        delta_sharpe_raw = torch.zeros(B, 1, device=state.device)
        cvar_raw = torch.zeros(B, 1, device=state.device)
        delta_component = torch.full_like(crisis_base_component, 0.5)
        cvar_component = torch.full_like(crisis_base_component, 0.5)

        if self.has_dsr_cvar and state.size(1) >= 2:
            # DSR/CVaR가 state에 포함된 경우 (reward_type='dsr_cvar')
            delta_sharpe_raw = state[:, -2:-1]  # [B, 1] - DSR bonus (원본 스케일 ~0.1)
            cvar_raw = state[:, -1:]  # [B, 1] - CVaR value (원본 스케일 ~0.01)

            _update_running_stats(delta_sharpe_raw, self.sharpe_mean, self.sharpe_var)
            _update_running_stats(cvar_raw, self.cvar_mean, self.cvar_var)

            sharpe_mean = self.sharpe_mean.to(state.device)
            sharpe_std = torch.sqrt(self.sharpe_var.to(state.device) + self.stats_eps)
            cvar_mean = self.cvar_mean.to(state.device)
            cvar_std = torch.sqrt(self.cvar_var.to(state.device) + self.stats_eps)

            sharpe_z = (delta_sharpe_raw - sharpe_mean) / sharpe_std
            cvar_z = (cvar_raw - cvar_mean) / cvar_std

            delta_component = torch.sigmoid(-self.k_s * sharpe_z)
            cvar_component = torch.sigmoid(self.k_c * cvar_z)

        crisis_raw = (
            self.w_r * crisis_base_component
            + self.w_s * delta_component
            + self.w_c * cvar_component
        )

        if self.training:
            with torch.no_grad():
                flat_raw = crisis_raw.detach().view(-1)
                if flat_raw.numel() > 0:
                    median = flat_raw.median()
                    mad = torch.median(torch.abs(flat_raw - median))
                    scale_update = torch.clamp(mad * 1.4826, min=self.stats_eps)
                    self.crisis_robust_loc.copy_(median.unsqueeze(0))
                    self.crisis_robust_scale.copy_(scale_update.unsqueeze(0))

        robust_loc = self.crisis_robust_loc.to(state.device)
        robust_scale = self.crisis_robust_scale.to(state.device)
        crisis_raw = (crisis_raw - robust_loc) / (robust_scale + self.stats_eps)

        if self.training and self.bias_warm_started.item() < 0.5:
            with torch.no_grad():
                mean_raw = crisis_raw.detach().mean()
                temp = self.crisis_temperature.to(state.device)
                temp_clamped = torch.clamp(temp, min=self.stats_eps)
                target = mean_raw.new_tensor(self.p_star)
                target = torch.clamp(target, self.stats_eps, 1 - self.stats_eps)
                target_logit = torch.log(target / (1 - target))
                new_bias = mean_raw - temp_clamped * target_logit
                self.crisis_bias.copy_(new_bias)
                self.bias_warm_started.fill_(1.0)

        bias = self.crisis_bias.to(state.device)
        temperature = self.crisis_temperature.to(state.device)
        crisis_affine = (crisis_raw - bias) / (temperature + self.stats_eps)
        crisis_level_pre_guard = torch.sigmoid(crisis_affine)

        p_hat = crisis_level_pre_guard.detach().mean()

        if self.training:
            with torch.no_grad():
                self.crisis_step.add_(1.0)
                eta_b_now = self._current_eta_b()
                self.crisis_bias.add_(eta_b_now * (p_hat - self.p_star))
                temperature_update = 1.0 + self.eta_T * (p_hat - self.p_star)
                new_temperature = torch.clamp(
                    self.crisis_temperature * temperature_update,
                    min=self.temperature_min,
                    max=self.temperature_max,
                )
                self.crisis_temperature.copy_(new_temperature)

        guard_rate = self._current_guard_rate()
        guard_target = crisis_level_pre_guard.new_full(
            crisis_level_pre_guard.shape, self.crisis_target
        )
        crisis_level = torch.clamp(
            crisis_level_pre_guard
            + guard_rate * (guard_target - crisis_level_pre_guard),
            min=0.0,
            max=1.0,
        )

        self._update_hysteresis_thresholds(crisis_level_pre_guard)
        hysteresis_up_val, hysteresis_down_val = self._current_hysteresis_thresholds()
        hysteresis_up_tensor = crisis_level_pre_guard.new_tensor(hysteresis_up_val)
        hysteresis_down_tensor = crisis_level_pre_guard.new_tensor(hysteresis_down_val)

        prev_regime_flag = (self.crisis_prev_regime > 0.5).float()
        if prev_regime_flag.item() >= 0.5:
            crisis_regime = torch.where(
                crisis_level < hysteresis_down_tensor,
                torch.zeros_like(crisis_level),
                torch.ones_like(crisis_level),
            )
        else:
            crisis_regime = torch.where(
                crisis_level > hysteresis_up_tensor,
                torch.ones_like(crisis_level),
                torch.zeros_like(crisis_level),
            )

        with torch.no_grad():
            new_prev = (crisis_regime.mean() > 0.5).float()
            self.crisis_prev_regime.copy_(new_prev.view(1))

        delta_sharpe = delta_sharpe_raw
        cvar = cvar_raw

        # --- 2단계: 에피토프 인코딩 ---
        E = self.epitope_encoder(state).view(B, self.m, self.emb_dim)  # [B, m, D]

        # --- 3단계: 프로토타입 확장 ---
        K = self.proto_keys.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]

        # --- 4단계: 프로토타입 적합도 설정 ---
        if fitness is None:
            # Critic이 없으면 프로토타입 적합도를 균등 분포로 설정한다.
            fitness = torch.ones(B, self.M, device=state.device)

        # --- 5단계: IRT 연산 수행 ---
        w_prev_batch = self.w_prev.expand(B, -1)  # [B, M]

        # Sharpe 추세 변화를 α 조정 단계에 전달한다.
        delta_sharpe_tensor = delta_sharpe

        if self.alpha_crisis_source == "pre_guard":
            crisis_for_alpha = crisis_level_pre_guard
        else:
            crisis_for_alpha = crisis_level

        w, P, irt_debug = self.irt(
            E=E,
            K=K,
            danger=danger_embed,
            w_prev=w_prev_batch,
            fitness=fitness,
            crisis_level=crisis_for_alpha,
            delta_sharpe=delta_sharpe_tensor,
            proto_conf=None,
        )

        # --- 6단계: Dirichlet 혼합 정책 계산 ---
        # 각 프로토타입의 concentration을 계산한다.
        # Tanh 출력 [-1, 1]을 확장하여 비대칭 bias를 적용하고 0.02 이상으로 클램프한다.
        concentrations_raw = torch.stack(
            [self.decoders[j](K[:, j, :]) for j in range(self.M)], dim=1
        )  # [B, M, A] - Tanh 출력은 [-1, 1] 범위에 위치한다.

        # Softplus와 비대칭 bias로 희소한 포트폴리오를 유도한다.
        concentrations = torch.nn.functional.softplus(concentrations_raw * 2.0) + 0.02
        proto_bias = torch.linspace(
            -0.5, 0.5, steps=self.M, device=concentrations.device, dtype=concentrations.dtype
        ).view(1, self.M, 1)
        concentrations = concentrations + 0.05 * proto_bias
        concentrations = torch.clamp(concentrations, min=0.02, max=self.dirichlet_max)

        # IRT 가중치로 혼합
        mixed_conc = torch.einsum("bm,bma->ba", w, concentrations)  # [B, A]

        if deterministic:
            # 결정적 모드: 혼합된 농도를 로그릿으로 보고 온도 조절 소프트맥스를 적용한다.
            # 온도가 낮을수록 집중도가 높아지고, 높을수록 분산된 행동이 만들어진다.
            action = F.softmax(mixed_conc / self.action_temp, dim=-1)
        else:
            # 확률적 모드: 클램프된 농도를 Dirichlet 파라미터로 사용해 행동을 샘플링한다.
            # α < 1이면 희소한 분포, α = 1이면 균등, α > 1이면 균등 인근에 모인다.
            # 수치 안정성을 위해 [dirichlet_min, dirichlet_max] 범위로 제한한다.
            mixed_conc_clamped = torch.clamp(
                mixed_conc, min=self.dirichlet_min, max=self.dirichlet_max
            )
            dist = torch.distributions.Dirichlet(mixed_conc_clamped)
            action = dist.sample()

        # Simplex 보장 (수치 안정성)
        action = torch.clamp(action, min=0.0, max=1.0)
        action = action / (action.sum(dim=-1, keepdim=True) + 1e-8)

        # --- 7단계: EMA 기반 w_prev 업데이트 ---
        if self.training:
            with torch.no_grad():
                self.w_prev = self.ema_beta * self.w_prev + (
                    1 - self.ema_beta
                ) * w.detach().mean(dim=0, keepdim=True)

        # --- 8단계: 해석 정보 수집 ---
        mixed_conc_det = mixed_conc.detach()
        mixed_conc_clamped_det = (
            mixed_conc_clamped.detach() if not deterministic else mixed_conc_det
        )

        info = {
            "w": w.detach(),  # [B, M] - 프로토타입 가중치
            "P": P.detach(),  # [B, m, M] - 수송 계획
            "crisis_level": crisis_level.detach(),  # [B, 1]
            "crisis_level_pre_guard": crisis_level_pre_guard.detach(),  # [B, 1]
            "alpha_crisis_input": crisis_for_alpha.detach(),  # [B, 1]
            "crisis_raw": crisis_raw.detach(),  # [B, 1]
            "crisis_bias": self.crisis_bias.detach().clone(),  # [1]
            "crisis_temperature": self.crisis_temperature.detach().clone(),  # [1]
            "crisis_prev_regime": self.crisis_prev_regime.detach().clone(),  # [1]
            "crisis_base_component": crisis_base_component.detach(),  # [B, 1]
            "delta_component": delta_component.detach(),  # [B, 1]
            "cvar_component": cvar_component.detach(),  # [B, 1]
            "crisis_regime": crisis_regime.detach(),  # [B, 1]
            "crisis_guard_rate": torch.tensor(guard_rate, device=state.device),
            "bias_warm_started": self.bias_warm_started.detach().clone(),
            "crisis_types": z.detach(),  # [B, K]
            "fitness": fitness.detach(),  # [B, M]
            "hysteresis_up": torch.tensor(hysteresis_up_val, device=state.device),
            "hysteresis_down": torch.tensor(hysteresis_down_val, device=state.device),
            # IRT 분해 정보 (시각화용)
            "w_rep": irt_debug["w_rep"].detach(),  # [B, M] - Replicator 출력
            "w_ot": irt_debug["w_ot"].detach(),  # [B, M] - OT 출력
            "cost_matrix": irt_debug["cost_matrix"].detach(),  # [B, m, M]
            "eta": irt_debug["eta"].detach(),  # [B, 1] - 위기 적응형 학습률
            "alpha_c": irt_debug[
                "alpha_c"
            ].detach(),  # [B, 1] - 동적으로 조정된 OT-Replicator 혼합 비율
            "alpha_c_raw": irt_debug.get(
                "alpha_c_raw", irt_debug["alpha_c"]
            ).detach(),  # [B, 1] - 클램프 전 α 값
            "alpha_c_prev": irt_debug.get(
                "alpha_c_prev", irt_debug["alpha_c"]
            ).detach(),  # [B, 1] - 직전 배치의 α
            "alpha_c_star": irt_debug.get(
                "alpha_c_star", irt_debug["alpha_c"]
            ).detach(),  # [B, 1] - 노이즈 적용 전 목표 α
            "alpha_c_decay_factor": irt_debug.get(
                "alpha_c_decay_factor", torch.ones_like(irt_debug["alpha_c"])
            ).detach(),  # [B, 1] - 방향성 감쇠 계수
            "pct_clamped_min": irt_debug.get(
                "pct_clamped_min", 0.0
            ),  # α가 하한에 도달한 비율
            "pct_clamped_max": irt_debug.get(
                "pct_clamped_max", 0.0
            ),  # α가 상한에 도달한 비율
            "alpha_candidate": irt_debug.get(
                "alpha_candidate", irt_debug["alpha_c"]
            ).detach(),
            "alpha_state": irt_debug.get(
                "alpha_state_buffer",
                self.irt.alpha_state.detach().clone()
                if hasattr(self.irt, "alpha_state")
                else torch.zeros(1, 1, device=state.device),
            ).detach(),
            "alpha_feedback_gain": irt_debug.get(
                "alpha_feedback_gain",
                torch.tensor(getattr(self.irt, "alpha_feedback_gain", 0.0), device=state.device),
            ).detach(),
            "alpha_feedback_bias": irt_debug.get(
                "alpha_feedback_bias",
                torch.tensor(getattr(self.irt, "alpha_feedback_bias", 0.0), device=state.device),
            ).detach(),
            "directional_decay_min": irt_debug.get(
                "directional_decay_min",
                torch.tensor(getattr(self.irt, "directional_decay_min", 0.0), device=state.device),
            ).detach(),
            # Dirichlet 정보 (로그 확률 계산용)
            "concentrations": concentrations.detach(),  # [B, M, A] - 프로토타입별 concentration
            "mixed_conc": mixed_conc_det,  # [B, A] - 혼합된 concentration (원본)
            "mixed_conc_clamped": mixed_conc_clamped_det,  # [B, A]
            # 프로토타입 디코더의 통계 정보를 기록한다.
            "concentrations_mean": concentrations.mean(
                dim=0
            ).detach(),  # [M, A] - 프로토타입별 평균
            "concentrations_std": (
                concentrations.std(dim=0, unbiased=False).detach()
                if B > 1
                else torch.zeros_like(concentrations[0]).detach()
            ),
            "mixed_conc_mean": mixed_conc.mean().detach(),  # 스칼라 - 혼합 후 평균
            "mixed_conc_std": (
                mixed_conc.std(unbiased=False).detach()
                if mixed_conc.numel() > 1
                else mixed_conc.new_tensor(0.0).detach()
            ),
            # 위기 감지와 입력 신호 통계를 함께 제공한다.
            "crisis_base": crisis_base_sigmoid.detach(),  # [B, 1] - T-Cell 기본 위기 신호 (sigmoid)
            "crisis_base_raw": crisis_affine_raw.detach(),  # [B, 1] - T-Cell affine
            "crisis_robust_loc": self.crisis_robust_loc.detach().to(state.device),
            "crisis_robust_scale": self.crisis_robust_scale.detach().to(state.device),
            "hysteresis_quantile": torch.tensor(self.hysteresis_quantile, device=state.device),
            "tech_indicator_count": torch.tensor(self.tech_indicator_count, device=state.device),
            "tech_indicator_count_active": torch.tensor(indicator_blocks, device=state.device),
            "has_dsr_cvar": torch.tensor(1 if self.has_dsr_cvar else 0, device=state.device),
            "stock_dim": torch.tensor(self.stock_dim, device=state.device),
            "delta_sharpe": (
                delta_sharpe.detach()
                if state.size(1) >= self.state_dim - 2
                else torch.zeros(B, 1, device=state.device)
            ),
            "cvar": (
                cvar.detach()
                if state.size(1) >= self.state_dim - 2
                else torch.zeros(B, 1, device=state.device)
            ),
        }

        if retain_grad:
            info["mixed_conc_grad"] = mixed_conc
            info["mixed_conc_clamped_grad"] = (
                mixed_conc_clamped if not deterministic else mixed_conc
            )
            info["action_temp"] = torch.tensor(self.action_temp, device=state.device)

        return action, info
