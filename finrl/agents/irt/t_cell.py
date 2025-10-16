# finrl/agents/irt/t_cell.py
# 시장 특징을 입력받아 위기 타입 점수와 공자극 임베딩을 추정하는 경량 T-Cell 모듈을 정의한다.

"""
경량 T-Cell 위기 감지기

이 모듈은 Isolation Forest와 같은 복잡한 기법을 제거하고,
단일 다층 퍼셉트론으로 위기 타입 점수(z), 공자극 임베딩(d), 위기 레벨(c)을 동시에 출력한다.
온라인 정규화를 통해 학습 중 수집되는 통계값을 안정적으로 유지한다.

출력:
- z: 위기 타입별 점수 [B, K]
- d: 공자극 임베딩 [B, D]
- crisis_affine: 가열 직전의 선형 조합 값 [B, 1]
- crisis_base: 시그모이드 스케일의 위기 확률 [B, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TCellMinimal(nn.Module):
    """시장 특성을 바탕으로 위기 수준과 공자극 임베딩을 추정하는 경량 T-Cell"""

    def __init__(
        self, in_dim: int, emb_dim: int = 128, n_types: int = 4, momentum: float = 0.99
    ):
        """
        Args:
            in_dim: 입력 차원 (시장 특성, 예: 12)
            emb_dim: 공자극 임베딩 차원
            n_types: 위기 타입 수 (변동성, 유동성, 상관관계, 시스템)
            momentum: 온라인 정규화에 사용되는 모멘텀 계수
        """
        super().__init__()

        self.n_types = n_types
        self.emb_dim = emb_dim
        self.momentum = momentum

        # 단일 인코더 (효율성)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_types + emb_dim),
        )

        # 온라인 정규화 통계 (학습 중 업데이트)
        self.register_buffer("mu", torch.zeros(n_types))
        self.register_buffer("sigma", torch.ones(n_types))
        self.register_buffer("count", torch.zeros(1))

        # 위기 타입별 가중치 (학습 가능)
        self.alpha = nn.Parameter(torch.ones(n_types) / n_types)

    def forward(
        self, features: torch.Tensor, update_stats: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: 시장 특성 [B, F]
            update_stats: 통계 업데이트 여부 (학습=True, 평가=False)

        Returns:
            z: 위기 타입 점수 [B, K]
            d: 공자극 임베딩 [B, D]
            crisis_affine: 가열 직전 선형 조합 값 [B, 1]
            crisis_base: 시그모이드 스케일 위기 확률 [B, 1]
        """
        h = self.encoder(features)  # [B, K+D]

        # 분리
        z = h[:, : self.n_types]  # [B, K]
        d = h[:, self.n_types :]  # [B, D]

        # 온라인 정규화 (학습 시, batch > 1)
        if update_stats and self.training and z.size(0) > 1:
            with torch.no_grad():
                batch_mu = z.mean(dim=0)
                batch_sigma = z.std(dim=0) + 1e-6

                # EMA 업데이트
                self.mu = self.momentum * self.mu + (1 - self.momentum) * batch_mu
                self.sigma = (
                    self.momentum * self.sigma + (1 - self.momentum) * batch_sigma
                )
                self.count += 1

        # 표준화
        z_std = (z - self.mu) / (self.sigma + 1e-6)  # [B, K]

        # 가중 합산 → 시그모이드 (0-1 범위)
        alpha_norm = F.softmax(self.alpha, dim=0)  # [K]
        crisis_affine = (z_std * alpha_norm).sum(dim=-1, keepdim=True)  # [B, 1]
        crisis_base = torch.sigmoid(crisis_affine)  # [B, 1]

        return z, d, crisis_affine, crisis_base
