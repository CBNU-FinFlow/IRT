# finrl/agents/irt/irt_operator.py
# Sinkhorn 기반의 IRT 연산자와 면역 비용 설계를 정의한다.

"""
IRT (Immune Replicator Transport) 연산자

이 모듈은 Optimal Transport와 Replicator Dynamics를 결합하여
포트폴리오 프로토타입 혼합 가중치를 계산하는 연산자를 구현한다.

핵심 식:
w_t = (1-α)·Replicator(w_{t-1}, f_t) + α·Transport(E_t, K, C_t)

주요 구성 요소:
- Sinkhorn: 엔트로피 정규화 최적수송 알고리즘
- 면역 비용 함수: 위험 신호·자기-내성·프로토타입 과신도를 포함한 비용 행렬
- 동적 혼합 비율 α: 위기 레벨과 Sharpe 추세에 따라 조정되는 혼합 계수
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

class Sinkhorn(nn.Module):
    """
    엔트로피 정규화 최적수송을 계산하는 Sinkhorn 반복기

    목적 함수:
        min_{P∈U(u,v)} <P, C> + ε · KL(P || u v^T)
    여기서 U(u, v)는 주어진 주변 분포를 만족하는 수송 행렬 집합이다.
    """

    def __init__(self, max_iters: int = 10, eps: float = 0.10, tol: float = 1e-3):
        super().__init__()
        self.max_iters = max_iters
        self.eps = eps
        self.tol = tol

    def forward(self, C: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            C: 비용 행렬 [B, m, M]
            u: 소스 분포 [B, m, 1]
            v: 타겟 분포 [B, 1, M]

        Returns:
            P: 수송 계획 [B, m, M]
        """
        B, m, M = C.shape

        # 로그 공간에서 연산하여 수치 안정성을 확보한다.
        log_K = -C / (self.eps + 1e-8)
        log_u = torch.log(u + 1e-8)
        log_v = torch.log(v + 1e-8)

        log_a = torch.zeros_like(log_u)
        log_b = torch.zeros_like(log_v)

        # Sinkhorn 반복
        for iter_idx in range(self.max_iters):
            log_a_prev = log_a.clone()

            log_a = log_u - torch.logsumexp(log_K + log_b, dim=2, keepdim=True)
            log_b = log_v - torch.logsumexp(log_K + log_a, dim=1, keepdim=True)

            # 조기 종료 (수렴 체크)
            if iter_idx > 0:
                err = torch.abs(log_a - log_a_prev).max()
                if err < self.tol:
                    break

        # 수송 계획 계산
        P = torch.exp(log_a + log_K + log_b)

        # 수치 안정성을 위해 수송 행렬 값을 0~1 범위로 제한한다.
        P = torch.clamp(P, min=0.0, max=1.0)

        return P

class IRT(nn.Module):
    """
    Immune Replicator Transport Operator

    핵심 혁신:
    1. OT: 구조적 매칭 (현재 상태 ↔ 프로토타입)
    2. Replicator: 시간 메모리 (과거 성공 전략 선호)
    3. 면역 신호: 비용 함수에 도메인 지식 내장

    수학적 정의:
    C_ij = d(e_i,k_j) - γ<e_i,d_t> + λ[tolerance] + ρ[checkpoint]
    P* = Sinkhorn(C, u, v)
    w_tilde ∝ w_{t-1}·exp(η(c)[f - \\bar{f}])
    w_t = (1-α)·w_tilde + α·P*1_m
    """

    def __init__(self,
                 emb_dim: int,
                 m_tokens: int = 6,
                 M_proto: int = 8,
                 eps: float = 0.03,  # Sinkhorn 반복에서 사용하는 엔트로피 계수
                 alpha: float = 0.45,  # 평시 기준의 OT 혼합 상한
                 alpha_min: float = 0.05,  # 위기 시 Replicator 비중을 확보하기 위한 α 하한
                 alpha_max: Optional[float] = 0.55,  # OT 혼합 비율의 상한 (None이면 alpha 사용)
                 gamma: float = 0.90,  # 공자극 내적을 비용에서 차감할 때 사용하는 가중치
                 lambda_tol: float = 2.0,
                 rho: float = 0.3,
                 eta_0: float = 0.05,
                 eta_1: float = 0.12,  # 위기 레벨에 따라 추가되는 Replicator 학습률 계수
                 alpha_update_rate: float = 0.95,
                 alpha_feedback_gain: float = 0.10,
                 alpha_feedback_bias: float = 0.0,
                 directional_decay_min: float = 0.0,
                 alpha_noise_std: float = 0.0,
                 kappa: float = 1.0,
                 eps_tol: float = 0.1,
                 n_self_sigs: int = 4,
                 max_iters: int = 30,
                 tol: float = 1e-3,
                 replicator_temp: float = 0.4):
        super().__init__()

        self.emb_dim = emb_dim
        self.m = m_tokens
        self.M = M_proto
        self.alpha = alpha  # 후진 호환

        # alpha_min이 극단적으로 작을 때는 수렴이 느려질 수 있으므로 경고만 출력한다.
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max if alpha_max is not None else alpha

        # 경고만 표시 (강제하지 않음)
        if alpha_min < 0.03:
            print(f"[IRT] Warning: alpha_min {alpha_min:.3f} is very low (< 0.03), may cause instability")

        # 비용 함수 가중치
        self.gamma = gamma          # 공자극 가중치
        self.lambda_tol = lambda_tol  # 내성 가중치
        self.rho = rho              # 체크포인트 가중치
        self.kappa = kappa          # 내성 게인
        self.eps_tol = eps_tol      # 내성 임계값

        # 위기 가열 메커니즘
        self.eta_0 = eta_0          # 기본 학습률
        self.eta_1 = eta_1          # 위기 시 증가량

        self.alpha_update_rate = float(alpha_update_rate)
        self.alpha_feedback_gain = float(alpha_feedback_gain)
        self.alpha_feedback_bias = float(alpha_feedback_bias)
        self.directional_decay_min = float(max(min(directional_decay_min, 1.0), 0.0))
        self.alpha_noise_std = float(max(alpha_noise_std, 0.0))

        # Replicator 온도 (균등 혼합 고착 해제)
        self.replicator_temp = replicator_temp

        # 학습 가능한 마할라노비스 메트릭
        # M = L^T L (positive definite 보장)
        self.metric_L = nn.Parameter(torch.eye(emb_dim))

        # 자기-내성 서명 (학습 가능)
        self.self_sigs = nn.Parameter(torch.randn(n_self_sigs, emb_dim) * 0.1)

        # Sinkhorn 알고리즘
        self.sinkhorn = Sinkhorn(max_iters=max_iters, eps=eps, tol=tol)

        init_alpha = float(self.alpha_max)
        self.register_buffer("alpha_state", torch.full((1, 1), init_alpha))

    def _mahalanobis_distance(self, E: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        학습 가능한 마할라노비스 거리

        d_M(x,y) = sqrt((x-y)^T M (x-y)), M = L^T L
        """
        M = self.metric_L.T @ self.metric_L  # [D, D]
        M = M.to(E.device)  # 디바이스 동기화

        diff = E.unsqueeze(2) - K.unsqueeze(1)  # [B, m, M, D]

        # (x-y)^T M (x-y) = sum_ij (x-y)_i M_ij (x-y)_j
        mahal_sq = torch.einsum('bmnd,de,bmne->bmn', diff, M, diff)
        mahal = torch.sqrt(torch.clamp(mahal_sq, min=1e-8))

        return mahal  # [B, m, M]

    def _cost_matrix(self,
                     E: torch.Tensor,
                     K: torch.Tensor,
                     danger: torch.Tensor,
                     proto_conf: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        면역학적 비용 행렬 구성

        C_ij = distance - γ·co_stim + λ·tolerance + ρ·checkpoint

        Args:
            E: 에피토프 [B, m, D]
            K: 프로토타입 [B, M, D]
            danger: 공자극 임베딩 [B, D]
            proto_conf: 프로토타입 과신도 [B, 1, M] (optional)
        """
        B, m, D = E.shape
        M = K.shape[1]

        # 1. 기본 거리
        dist = self._mahalanobis_distance(E, K)  # [B, m, M]

        # 2. 공자극 (Co-stimulation)
        # 위험 신호와 정렬된 에피토프 선호
        co_stim = torch.einsum('bmd,bd->bm', E, danger).unsqueeze(2)  # [B, m, 1]

        # 3. 음성 선택 (Tolerance)
        # 자기-내성 서명과 유사한 에피토프 억제
        E_norm = F.normalize(E, dim=-1)  # [B, m, D]
        sig_norm = F.normalize(self.self_sigs.to(E.device), dim=-1)  # [S, D]

        cos_sim = E_norm @ sig_norm.T  # [B, m, S]
        worst_match = cos_sim.max(dim=-1, keepdim=True)[0]  # [B, m, 1]

        tolerance_penalty = torch.relu(
            self.kappa * worst_match - self.eps_tol
        )  # [B, m, 1]

        # 4. 체크포인트 억제 (Checkpoint)
        # 과신하는 프로토타입 억제
        if proto_conf is None:
            proto_conf = torch.zeros(B, 1, M, device=E.device)

        # 5. 종합 비용
        C = (
            dist
            - self.gamma * co_stim
            + self.lambda_tol * tolerance_penalty
            + self.rho * proto_conf
        )

        return C  # [B, m, M]

    def forward(self,
                E: torch.Tensor,
                K: torch.Tensor,
                danger: torch.Tensor,
                w_prev: torch.Tensor,
                fitness: torch.Tensor,
                crisis_level: torch.Tensor,
                delta_sharpe: torch.Tensor,
                proto_conf: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        IRT 연산자 forward pass

        Args:
            E: 에피토프 [B, m, D]
            K: 프로토타입 [B, M, D]
            danger: 공자극 임베딩 [B, D]
            w_prev: 이전 혼합 가중치 [B, M]
            fitness: 프로토타입 적합도 [B, M]
            crisis_level: 위기 레벨 [B, 1]
            delta_sharpe: DSR bonus [B, 1] (Sharpe gradient feedback용)
            proto_conf: 프로토타입 과신도 [B, 1, M]

        Returns:
            w: 새 혼합 가중치 [B, M]
            P: 수송 계획 [B, m, M] (해석용)
            debug_info: IRT 분해 정보 (시각화용)
        """
        B, m, D = E.shape
        M = K.shape[1]

        # --- 1단계: Optimal Transport 매칭 ---
        u = torch.full((B, m, 1), 1.0/m, device=E.device)
        v = torch.full((B, 1, M), 1.0/M, device=E.device)

        C = self._cost_matrix(E, K, danger, proto_conf)
        P = self.sinkhorn(C, u, v)  # [B, m, M]

        # OT 마진 (프로토타입별 수송 질량)
        p_mass = P.sum(dim=1)  # [B, M]

        # --- 2단계: Replicator 기반 가중치 적응 ---
        # 위기 가열: η(c) = η_0 + η_1·c (NaN 방어)
        crisis_level_safe = torch.nan_to_num(crisis_level, nan=0.0)
        eta = self.eta_0 + self.eta_1 * crisis_level_safe  # [B, 1]

        # Advantage 계산
        baseline = (w_prev * fitness).sum(dim=-1, keepdim=True)  # [B, 1]
        advantage = fitness - baseline  # [B, M]

        # 자기-내성 페널티 (프로토타입도 검사)
        K_norm = F.normalize(K, dim=-1)  # [B, M, D]
        sig_norm = F.normalize(self.self_sigs.to(K.device), dim=-1)  # [S, D]

        proto_self_sim = (K_norm @ sig_norm.T).max(dim=-1)[0]  # [B, M]
        r_penalty = 0.5 * proto_self_sim

        # 로그 공간에서 계산한 Replicator 방정식
        log_w_prev = torch.log(w_prev + 1e-8)
        log_tilde_w = log_w_prev + eta * advantage - r_penalty

        # 온도 조절 소프트맥스로 Replicator 출력의 집중도를 제어한다.
        tilde_w = F.softmax(log_tilde_w / self.replicator_temp, dim=-1)  # [B, M]

        # --- 3단계: 위기 수준 기반 α(c) 계산 ---
        # α(c) = α_max + (α_min - α_max) · (1 - cos(πc)) / 2
        # c=0 → α=α_max, c=1 → α=α_min, 중간에서 민감도 최대
        pi_c = torch.tensor(torch.pi, device=crisis_level_safe.device) * torch.clamp(crisis_level_safe, 0.0, 1.0)
        alpha_c = self.alpha_max + (self.alpha_min - self.alpha_max) * (1 - torch.cos(pi_c)) / 2

        # --- 3단계 보강: Sharpe 기울기 피드백으로 α(c) 재조정 ---
        # delta_sharpe > 0 (성과 개선) → alpha_c 증가 → OT 비중 확대
        # delta_sharpe < 0 (성과 악화) → alpha_c 감소 → Replicator 비중 확대
        delta_sharpe_safe = torch.nan_to_num(delta_sharpe, nan=0.0)  # [B, 1]
        delta_tanh = torch.tanh(delta_sharpe_safe)
        alpha_c_raw = alpha_c * (1 + self.alpha_feedback_gain * delta_tanh) + self.alpha_feedback_bias * delta_tanh

        # 방향성 감쇠를 적용하여 α 변화량을 완화한다.
        alpha_star = torch.clamp(alpha_c_raw, min=self.alpha_min, max=self.alpha_max)
        if self.alpha_noise_std > 0 and self.training:
            noise = torch.randn_like(alpha_star) * self.alpha_noise_std
            alpha_star = torch.clamp(alpha_star + noise, min=self.alpha_min, max=self.alpha_max)
        alpha_prev = self.alpha_state.to(alpha_star.device).expand(B, 1)
        delta_raw = alpha_star - alpha_prev

        span = (self.alpha_max - self.alpha_min) + 1e-9
        pos = torch.clamp((alpha_prev - self.alpha_min) / span, 0.0, 1.0)
        dist_to_bound = torch.min(pos, 1.0 - pos)
        centered = torch.clamp(dist_to_bound * 2.0, min=0.0, max=1.0)
        p = 1.0
        decay_min = torch.tensor(
            self.directional_decay_min, device=delta_raw.device, dtype=delta_raw.dtype
        )
        decay_factor = decay_min + (1.0 - decay_min) * (centered ** p)

        delta = self.alpha_update_rate * delta_raw * decay_factor
        alpha_candidate = alpha_prev + delta
        alpha_c = torch.clamp(alpha_candidate, min=self.alpha_min, max=self.alpha_max)

        pct_clamped_min = (alpha_candidate < self.alpha_min).float().mean()
        pct_clamped_max = (alpha_candidate > self.alpha_max).float().mean()

        with torch.no_grad():
            updated_state = alpha_c.mean(dim=0, keepdim=True)
            self.alpha_state.copy_(updated_state.detach())
        # alpha_c: [B, 1]

        # --- 4단계: Replicator와 OT 가중치 결합 ---
        w = (1 - alpha_c) * tilde_w + alpha_c * p_mass

        # 정규화 (수치 안정성, NaN 방어)
        w = torch.nan_to_num(w, nan=1.0/self.M)  # NaN 시 균등 분포
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        w = torch.clamp(w, min=1e-6, max=1.0)
        w = w / w.sum(dim=-1, keepdim=True)  # 재정규화 (합=1)

        # --- 5단계: 해석 및 시각화를 위한 보조 정보 ---
        debug_info = {
            'w_rep': tilde_w,  # [B, M] - Replicator 경로에서 생성된 가중치
            'w_ot': p_mass,    # [B, M] - Optimal Transport 경로의 질량 합
            'cost_matrix': C,  # [B, m, M] - 면역학적 비용 행렬
            'eta': eta,        # [B, 1] - 위기 레벨에 반응하는 학습률
            'alpha_c': alpha_c,  # [B, 1] - 동적으로 조정된 OT-Replicator 혼합 비율
            # α 업데이트 과정을 해석하기 위한 세부 정보
            'alpha_c_raw': alpha_c_raw,  # [B, 1] - 클램프 적용 전 α 값
            'alpha_c_prev': alpha_prev.detach(),  # [B, 1] - 직전 배치에서 유지된 α
            'alpha_c_star': alpha_star,  # [B, 1] - 노이즈·클램프 이전의 목표 α
            'alpha_c_decay_factor': decay_factor,  # [B, 1] - 방향성 감쇠 계수
            'pct_clamped_min': pct_clamped_min,  # 스칼라 - α가 하한에 걸린 비율
            'pct_clamped_max': pct_clamped_max,  # 스칼라 - α가 상한에 걸린 비율
            'alpha_candidate': alpha_candidate.detach(),  # [B, 1] - 감쇠 적용 후 후보 α
            'alpha_state_buffer': self.alpha_state.detach().clone(),  # [1, 1] - 지연 업데이트 상태
            'alpha_feedback_gain': torch.tensor(self.alpha_feedback_gain, device=alpha_c.device),
            'alpha_feedback_bias': torch.tensor(self.alpha_feedback_bias, device=alpha_c.device),
            'directional_decay_min': torch.tensor(self.directional_decay_min, device=alpha_c.device),
        }

        return w, P, debug_info
