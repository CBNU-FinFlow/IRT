# finrl/meta/env_stock_trading/reward_functions.py

"""
리스크 민감 보상 함수 모듈

DSR(Differential Sharpe Ratio)와 CVaR 기반 보상을 결합해 Sharpe 비율과 꼬리 위험을 동시에 관리한다.

이론적 근거:
- DSR: Moody & Saffell (1998), "Learning to Trade via Direct RL"
- CVaR: Tamar et al. (2015), "Policy Gradient with CVaR Constraints"
- 온라인 추정: Welford (1962) 온라인 평균/분산 알고리즘
"""

import numpy as np
from collections import deque


class DifferentialSharpeRatio:
    """
    온라인 Differential Sharpe Ratio (DSR) 계산

    Sharpe의 시간 미분 근사를 통해 직접 Sharpe 상승 방향 그라디언트 제공.

    수학적 정의:
    - μ_t = β·μ_{t-1} + (1-β)·R_t (이동평균)
    - σ²_t = β·σ²_{t-1} + (1-β)·(R_t - μ_t)² (이동분산)
    - S_t = μ_t / σ_t (Sharpe)
    - ΔS_t ≈ z_t - (S_{t-1}/2)·z_t², z_t = (R_t - μ_{t-1}) / σ_{t-1}

    Args:
        beta: EMA 계수 (0.95 권장, 빠른 반응 속도 확보)
        epsilon: 분모 안정화 (σ=0 방지)
    """

    def __init__(self, beta: float = 0.95, epsilon: float = 1e-8):
        self.beta = beta
        self.epsilon = epsilon

        # 상태 변수
        self.mu = 0.0  # 평균 수익
        self.sigma_sq = 1.0  # 분산 (초기값 1.0)
        self.sharpe = 0.0  # 현재 Sharpe

    def update(self, return_t: float) -> float:
        """
        새 리턴 관찰 시 DSR 증분 계산

        Args:
            return_t: 시점 t의 로그수익 (예: log(V_t/V_{t-1}))

        Returns:
            dsr: Sharpe 증분 (보상 보너스로 사용)
        """
        # 이전 통계 저장
        mu_prev = self.mu
        sigma_prev = np.sqrt(self.sigma_sq + self.epsilon)

        # 온라인 평균 업데이트
        self.mu = self.beta * self.mu + (1.0 - self.beta) * return_t

        # 온라인 분산 업데이트 (Welford)
        delta = return_t - self.mu
        self.sigma_sq = self.beta * self.sigma_sq + (1.0 - self.beta) * delta**2

        # 현재 Sharpe
        sigma_curr = np.sqrt(self.sigma_sq + self.epsilon)
        self.sharpe = self.mu / sigma_curr

        # DSR 증분 계산
        if sigma_prev > self.epsilon:
            z = (return_t - mu_prev) / sigma_prev
            dsr = z - 0.5 * self.sharpe * z**2
        else:
            # 초기 단계 (통계 불안정)
            dsr = 0.0

        return dsr

    def reset(self):
        """에피소드 종료 시 상태 초기화"""
        self.mu = 0.0
        self.sigma_sq = 1.0
        self.sharpe = 0.0


class CVaREstimator:
    """
    온라인 Conditional Value at Risk (CVaR) 추정

    최근 H 스텝의 리턴 분포에서 α% 하위 평균 계산.
    꼬리 위험 제어를 통해 Sharpe 개선 및 downside 보호.

    수학적 정의:
    - VaR_α = percentile(returns, α)
    - CVaR_α = E[R | R ≤ VaR_α]

    Args:
        alpha: CVaR 분위수 (0.05 = 5% 하위)
        window: 추정 윈도우 크기 (스텝)
    """

    def __init__(self, alpha: float = 0.05, window: int = 50):
        self.alpha = alpha
        self.window = window
        self.buffer = deque(maxlen=window)

    def update(self, return_t: float) -> float:
        """
        새 리턴 관찰 시 CVaR 추정

        Args:
            return_t: 시점 t의 로그수익

        Returns:
            cvar: CVaR_α 추정값 (음수 = 손실)
        """
        self.buffer.append(return_t)

        # 초기 워밍업 (충분한 샘플 확보 전)
        if len(self.buffer) < self.window // 2:
            return 0.0

        # α% VaR/CVaR 계산
        sorted_returns = np.sort(self.buffer)
        cutoff_idx = max(1, int(self.alpha * len(sorted_returns)))
        cvar = np.mean(sorted_returns[:cutoff_idx])

        return cvar

    def reset(self):
        """에피소드 종료 시 버퍼 초기화"""
        self.buffer.clear()


class RiskSensitiveReward:
    r"""
    DSR와 CVaR를 결합해 로그수익, Sharpe 지향 보너스, 꼬리 위험 패널티를 동시에 반영한다.

    - EMA 기반 정규화로 DSR/CVaR의 스케일을 통일한다.
    - 보상은 [-0.01, 0.01] 범위로 클리핑하여 이상값을 억제한다.

    최종 보상: \\(r'_t = r_t + \lambda_S\,\hat{\Delta S}_t - \lambda_{\text{CVaR}}\,\hat{L}_t\\)
    여기서 \\(\hat{\Delta S}_t\\)와 \\(\hat{L}_t\\)는 EMA로 정규화된 DSR, CVaR 값이다.

    Args:
        lambda_dsr: DSR 가중치 (권장 0.1~0.2)
        lambda_cvar: CVaR 가중치 (권장 0.05~0.1)
        dsr_beta: DSR 이동평균 계수
        cvar_alpha: CVaR 분위수
        cvar_window: CVaR 추정 윈도우 길이
        normalization_window: EMA 정규화에 사용하는 윈도우 길이 (기본: 30)
    """

    def __init__(
        self,
        lambda_dsr: float = 0.1,
        lambda_cvar: float = 0.05,
        dsr_beta: float = 0.99,
        cvar_alpha: float = 0.05,
        cvar_window: int = 50,
        normalization_window: int = 30,
    ):
        self.lambda_dsr = lambda_dsr
        self.lambda_cvar = lambda_cvar

        self.dsr = DifferentialSharpeRatio(beta=dsr_beta)
        self.cvar = CVaREstimator(alpha=cvar_alpha, window=cvar_window)

        # EMA 기반 정규화 계수
        self.norm_window = normalization_window
        self.norm_beta = 1.0 - 1.0 / normalization_window

        # 각 신호의 평균 절댓값 (초기값 1.0)
        self.log_return_ema = 1.0
        self.dsr_ema = 1.0
        self.cvar_ema = 1.0

    def compute(self, basic_return: float) -> tuple[float, dict]:
        """
        리스크 민감 보상을 계산한다 (정규화 및 클리핑 포함).

        Args:
            basic_return: 기본 로그수익 (log(V_t/V_{t-1}))

        Returns:
            reward: 혼합 보상 (정규화 + 클리핑 적용)
            info: 디버깅 정보 (dsr, cvar, 정규화 상태)
        """
        # DSR 보너스
        dsr_bonus = self.dsr.update(basic_return)

        # CVaR 패널티
        cvar_value = self.cvar.update(basic_return)
        cvar_penalty = abs(cvar_value) if cvar_value < 0 else 0.0

        # 정규화 분모를 EMA 방식으로 업데이트한다.
        self.log_return_ema = (
            self.norm_beta * self.log_return_ema
            + (1.0 - self.norm_beta) * abs(basic_return)
        )
        self.dsr_ema = (
            self.norm_beta * self.dsr_ema + (1.0 - self.norm_beta) * abs(dsr_bonus)
        )
        self.cvar_ema = (
            self.norm_beta * self.cvar_ema
            + (1.0 - self.norm_beta) * abs(cvar_value)
        )

        # 신호 스케일을 통일하기 위해 EMA 값으로 정규화하고 0으로 나누지 않도록 오프셋을 더한다.
        norm_dsr = dsr_bonus / (self.dsr_ema + 1e-6)
        norm_cvar = cvar_penalty / (self.cvar_ema + 1e-6)

        # 혼합 보상 (정규화된 신호 사용)
        reward = basic_return + self.lambda_dsr * norm_dsr - self.lambda_cvar * norm_cvar

        # 보상은 [-0.01, 0.01] 범위로 제한한다.
        reward_clipped = np.clip(reward, -0.01, 0.01)

        # 디버깅 정보
        info = {
            "dsr_bonus": dsr_bonus,
            "cvar_value": cvar_value,
            "cvar_penalty": cvar_penalty,
            "sharpe_online": self.dsr.sharpe,
            # 정규화 및 클리핑 이전/이후 값을 함께 기록한다.
            "norm_dsr": norm_dsr,
            "norm_cvar": norm_cvar,
            "reward_pre_clip": reward,
            "reward_clipped": reward_clipped,
            "log_return_ema": self.log_return_ema,
            "dsr_ema": self.dsr_ema,
            "cvar_ema": self.cvar_ema,
        }

        return reward_clipped, info

    def reset(self):
        """에피소드 종료 시 내부 상태와 EMA를 모두 초기화한다."""
        self.dsr.reset()
        self.cvar.reset()

        # 정규화 분모 초기값 설정
        self.log_return_ema = 1.0
        self.dsr_ema = 1.0
        self.cvar_ema = 1.0


class AdaptiveRiskReward:
    """
    위기 민감형 위험 보상 함수

    로그 수익 + 적응형 Sharpe 보너스 + CVaR 패널티 + 거래 비용 패널티로 구성된다.

    수식:
    r_t = log(V_t/V_{t-1}) + κ_S(c)·ΔSharpe - κ_C(c)·CVaR - μ·turnover

    여기서:
    - log(V_t/V_{t-1}): 항상 non-zero gradient를 보장하는 base return
    - κ_S(c) = λ_S + g_S·c (Sharpe 보너스 게이팅, g_S < 0)
    - κ_C(c) = λ_C + g_C·c (CVaR 패널티 게이팅, g_C > 0 권장)
    - ΔSharpe: DSR (Differential Sharpe Ratio)
    - CVaR: 꼬리 위험 (음수)
    - turnover: 거래 비용 (실행가중 기반)

    특징:
    1. Base log-return이 항상 존재 → gradient vanishing 해결
    2. Crisis-aware κ → 위기 시 자동으로 risk-averse 강화
    3. Direct CVaR penalty → Risk-aware value learning

    Args:
        lambda_sharpe: ΔSharpe 기본 가중치 (κ_S, default: 0.20)
        lambda_cvar: CVaR penalty 기본 가중치 (κ_C, default: 0.40)
        lambda_turnover: Turnover penalty 가중치 (μ, default: 0.0)
        crisis_gain_sharpe: Sharpe 게이팅에 대한 위기 계수 g_S (default: -0.15)
        crisis_gain_cvar: CVaR 게이팅에 대한 위기 계수 g_C (default: 0.25)
        dsr_beta: DSR 이동평균 계수 (default: 0.92)
        cvar_alpha: CVaR 분위수 (default: 0.05)
        cvar_window: CVaR 추정 윈도우 (default: 40)
    """

    def __init__(
        self,
        lambda_sharpe: float = 0.20,
        lambda_cvar: float = 0.40,
        lambda_turnover: float = 0.0,
        crisis_gain_sharpe: float = -0.15,
        crisis_gain_cvar: float = 0.25,
        dsr_beta: float = 0.92,
        cvar_alpha: float = 0.05,
        cvar_window: int = 40,
        scale_beta: float = 0.98,
    ):
        self.lambda_sharpe_base = lambda_sharpe
        self.lambda_cvar_base = lambda_cvar
        self.lambda_cvar = lambda_cvar  # backward compatibility (logging)
        self.lambda_turnover = lambda_turnover
        self.crisis_gain_sharpe = crisis_gain_sharpe
        self.crisis_gain_cvar = crisis_gain_cvar

        # DSR and CVaR estimators
        self.dsr = DifferentialSharpeRatio(beta=dsr_beta)
        self.cvar = CVaREstimator(alpha=cvar_alpha, window=cvar_window)

        # Crisis level (updated externally via callback)
        self.crisis_level = 0.5  # Default: neutral

        # Turnover tracking
        self.prev_weights = None

        # Adaptive scaling (EMA of signal magnitudes)
        self.scale_beta = float(np.clip(scale_beta, 0.0, 0.999))
        self._delta_sharpe_scale = 1e-3
        self._cvar_scale = 1e-3
        self._norm_eps = 1e-8

    def set_crisis_level(self, crisis_level: float):
        """
        Policy callback에서 crisis_level을 업데이트

        Args:
            crisis_level: 0 (평시) ~ 1 (위기)
        """
        self.crisis_level = np.clip(crisis_level, 0.0, 1.0)

    def compute(
        self,
        basic_return: float,
        current_weights: np.ndarray = None
    ) -> tuple[float, dict]:
        """
        Adaptive Risk-Aware Reward를 계산한다.

        Args:
            basic_return: log(V_t/V_{t-1})
            current_weights: 현재 포트폴리오 가중치 (turnover 계산용, optional)

        Returns:
            reward: 최종 보상
            info: 디버깅 정보
        """
        # 1. Base log-return (항상 non-zero gradient)
        log_return = basic_return

        # 2. ΔSharpe (DSR)
        delta_sharpe = self.dsr.update(basic_return)

        # 3. CVaR (꼬리 위험)
        cvar_value = self.cvar.update(basic_return)
        cvar_penalty = abs(cvar_value) if cvar_value < 0 else 0.0

        # 4. Adaptive 게이팅 (Sharpe ↘, CVaR ↗ when crisis ↑)
        kappa_sharpe = max(
            0.0, self.lambda_sharpe_base + self.crisis_gain_sharpe * self.crisis_level
        )
        kappa_cvar = max(
            0.0, self.lambda_cvar_base + self.crisis_gain_cvar * self.crisis_level
        )

        # 5. Risk bonus (Sharpe contribution minus CVaR penalty)
        self._delta_sharpe_scale = (
            self.scale_beta * self._delta_sharpe_scale
            + (1.0 - self.scale_beta) * abs(delta_sharpe)
        )
        self._cvar_scale = (
            self.scale_beta * self._cvar_scale
            + (1.0 - self.scale_beta) * abs(cvar_penalty)
        )

        delta_sharpe_norm = delta_sharpe / (self._norm_eps + self._delta_sharpe_scale)
        cvar_norm = cvar_penalty / (self._norm_eps + self._cvar_scale)

        sharpe_term = kappa_sharpe * delta_sharpe_norm
        cvar_term = -kappa_cvar * cvar_norm
        risk_bonus = sharpe_term + cvar_term

        # 6. Turnover penalty
        turnover_penalty = 0.0
        if current_weights is not None and self.prev_weights is not None:
            # Turnover = 0.5 * Σ|w_t - w_{t-1}|
            turnover = 0.5 * np.sum(np.abs(current_weights - self.prev_weights))
            turnover_penalty = self.lambda_turnover * turnover

        # Update prev_weights
        if current_weights is not None:
            self.prev_weights = current_weights.copy()

        # 7. 최종 보상
        reward_pre_tanh = log_return + risk_bonus - turnover_penalty
        reward = np.tanh(reward_pre_tanh)

        # 디버깅 정보
        info = {
            "log_return": log_return,
            "delta_sharpe": delta_sharpe,
            "cvar_value": cvar_value,
            "cvar_penalty": cvar_penalty,
            "delta_sharpe_normalized": delta_sharpe_norm,
            "cvar_normalized": cvar_norm,
            "crisis_level": self.crisis_level,
            "kappa": kappa_sharpe,
            "kappa_sharpe": kappa_sharpe,
            "kappa_cvar": kappa_cvar,
            "risk_bonus": risk_bonus,
            "turnover_penalty": turnover_penalty,
            "sharpe_online": self.dsr.sharpe,
            "reward_pre_clip": reward_pre_tanh,  # backward compatibility
            "reward_total": reward,
            "components": {
                "log_return": log_return,
                "sharpe_term": sharpe_term,
                "cvar_term": cvar_term,
                "turnover": -turnover_penalty,
                "kappa_sharpe": kappa_sharpe,
                "kappa_cvar": kappa_cvar,
            },
        }

        return reward, info

    def reset(self):
        """에피소드 종료 시 상태 초기화"""
        self.dsr.reset()
        self.cvar.reset()
        self.crisis_level = 0.5
        self.prev_weights = None
        self._delta_sharpe_scale = 1e-3
        self._cvar_scale = 1e-3
