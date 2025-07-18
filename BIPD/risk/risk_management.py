"""
리스크 관리 시스템
포트폴리오 리스크 평가 및 제약 조건 관리
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class RiskManager:
    """
    포트폴리오 리스크 관리 시스템
    포지션 크기 제한, 집중도 관리, 터노버 제약 등을 처리
    """
    
    def __init__(self, 
                 max_position_size: float = 0.4,
                 max_turnover: float = 0.3,
                 min_position_size: float = 0.01,
                 max_concentration: float = 0.6):
        self.max_position_size = max_position_size
        self.max_turnover = max_turnover
        self.min_position_size = min_position_size
        self.max_concentration = max_concentration
        
        # 이전 포지션 추적
        self.previous_weights = None
        self.position_history = []
        self.turnover_history = []
        
        # 리스크 메트릭
        self.risk_metrics = {
            'position_violations': 0,
            'turnover_violations': 0,
            'concentration_violations': 0
        }
    
    def apply_constraints(self, 
                         raw_weights: np.ndarray,
                         current_prices: np.ndarray = None,
                         crisis_level: float = 0.0) -> np.ndarray:
        """
        포트폴리오 가중치에 제약 조건 적용
        
        Args:
            raw_weights: 원시 포트폴리오 가중치
            current_prices: 현재 가격 (선택사항)
            crisis_level: 위기 수준 (0-1)
            
        Returns:
            제약 조건이 적용된 포트폴리오 가중치
        """
        # 1. 기본 정규화
        weights = self._normalize_weights(raw_weights)
        
        # 2. 포지션 크기 제한
        weights = self._apply_position_limits(weights, crisis_level)
        
        # 3. 집중도 제한
        weights = self._apply_concentration_limits(weights)
        
        # 4. 터노버 제한
        if self.previous_weights is not None:
            weights = self._apply_turnover_limits(weights, crisis_level)
        
        # 5. 최소 포지션 크기 적용
        weights = self._apply_minimum_positions(weights)
        
        # 6. 최종 정규화
        weights = self._normalize_weights(weights)
        
        # 7. 히스토리 업데이트
        self._update_history(weights)
        
        return weights
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """가중치 정규화"""
        weights = np.maximum(weights, 0)  # 음수 제거 (롱 온리)
        weight_sum = np.sum(weights)
        
        if weight_sum > 0:
            return weights / weight_sum
        else:
            return np.ones(len(weights)) / len(weights)
    
    def _apply_position_limits(self, weights: np.ndarray, crisis_level: float) -> np.ndarray:
        """포지션 크기 제한 적용"""
        # 위기 시 포지션 크기 제한 강화
        dynamic_max = self.max_position_size * (1 - crisis_level * 0.3)
        
        # 제한 초과 포지션 식별
        violations = weights > dynamic_max
        
        if np.any(violations):
            self.risk_metrics['position_violations'] += 1
            
            # 초과 포지션 조정
            excess_weights = weights[violations] - dynamic_max
            weights[violations] = dynamic_max
            
            # 초과분을 다른 포지션에 재분배
            remaining_positions = ~violations
            if np.any(remaining_positions):
                redistribution = np.sum(excess_weights) / np.sum(remaining_positions)
                weights[remaining_positions] += redistribution
        
        return weights
    
    def _apply_concentration_limits(self, weights: np.ndarray) -> np.ndarray:
        """집중도 제한 적용"""
        # 상위 포지션 집중도 확인
        sorted_weights = np.sort(weights)[::-1]  # 내림차순
        
        # 상위 3개 포지션 집중도
        top3_concentration = np.sum(sorted_weights[:3])
        
        if top3_concentration > self.max_concentration:
            self.risk_metrics['concentration_violations'] += 1
            
            # 집중도 조정
            adjustment_factor = self.max_concentration / top3_concentration
            top3_indices = np.argsort(weights)[::-1][:3]
            
            # 상위 3개 포지션 축소
            weights[top3_indices] *= adjustment_factor
            
            # 나머지 포지션 증가
            remaining_indices = np.argsort(weights)[::-1][3:]
            if len(remaining_indices) > 0:
                boost_factor = (1 - self.max_concentration) / np.sum(weights[remaining_indices])
                weights[remaining_indices] *= boost_factor
        
        return weights
    
    def _apply_turnover_limits(self, weights: np.ndarray, crisis_level: float) -> np.ndarray:
        """터노버 제한 적용"""
        # 현재 터노버 계산
        current_turnover = np.sum(np.abs(weights - self.previous_weights))
        
        # 위기 시 터노버 제한 완화
        dynamic_max_turnover = self.max_turnover * (1 + crisis_level * 0.5)
        
        if current_turnover > dynamic_max_turnover:
            self.risk_metrics['turnover_violations'] += 1
            
            # 터노버 조정
            adjustment_factor = dynamic_max_turnover / current_turnover
            weight_diff = weights - self.previous_weights
            adjusted_diff = weight_diff * adjustment_factor
            weights = self.previous_weights + adjusted_diff
        
        return weights
    
    def _apply_minimum_positions(self, weights: np.ndarray) -> np.ndarray:
        """최소 포지션 크기 적용"""
        # 최소 크기 미만 포지션 제거
        small_positions = weights < self.min_position_size
        
        if np.any(small_positions):
            # 작은 포지션을 0으로 설정
            removed_weight = np.sum(weights[small_positions])
            weights[small_positions] = 0
            
            # 남은 포지션에 재분배
            remaining_positions = weights > 0
            if np.any(remaining_positions):
                weights[remaining_positions] += removed_weight / np.sum(remaining_positions)
        
        return weights
    
    def _update_history(self, weights: np.ndarray):
        """히스토리 업데이트"""
        self.position_history.append({
            'timestamp': datetime.now().isoformat(),
            'weights': weights.copy(),
            'max_position': np.max(weights),
            'concentration': np.sum(np.sort(weights)[::-1][:3])
        })
        
        # 터노버 계산 및 저장
        if self.previous_weights is not None:
            turnover = np.sum(np.abs(weights - self.previous_weights))
            self.turnover_history.append({
                'timestamp': datetime.now().isoformat(),
                'turnover': turnover
            })
        
        self.previous_weights = weights.copy()
    
    def calculate_portfolio_risk(self, 
                               weights: np.ndarray,
                               returns: np.ndarray,
                               window: int = 252) -> Dict[str, float]:
        """
        포트폴리오 리스크 계산
        
        Args:
            weights: 포트폴리오 가중치
            returns: 수익률 데이터
            window: 계산 윈도우
            
        Returns:
            리스크 메트릭
        """
        if len(returns) < window:
            window = len(returns)
        
        recent_returns = returns[-window:]
        portfolio_returns = np.dot(recent_returns, weights)
        
        # 기본 리스크 메트릭
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        # 최대 손실 (Maximum Drawdown)
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # 샤프 비율
        excess_returns = portfolio_returns - 0.02/252  # 2% 무위험 수익률 가정
        sharpe_ratio = np.mean(excess_returns) / np.std(portfolio_returns) * np.sqrt(252)
        
        # 집중도 리스크
        concentration_risk = np.sum(weights**2)  # HHI
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'var_99': var_99,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'concentration_risk': concentration_risk,
            'max_position': np.max(weights),
            'num_positions': np.sum(weights > 0.01)
        }
    
    def get_risk_report(self) -> Dict[str, Any]:
        """리스크 관리 보고서 생성"""
        avg_turnover = np.mean([h['turnover'] for h in self.turnover_history]) if self.turnover_history else 0
        
        return {
            'risk_metrics': self.risk_metrics.copy(),
            'constraints': {
                'max_position_size': self.max_position_size,
                'max_turnover': self.max_turnover,
                'min_position_size': self.min_position_size,
                'max_concentration': self.max_concentration
            },
            'current_stats': {
                'avg_turnover': avg_turnover,
                'total_positions': len(self.position_history),
                'violation_rate': sum(self.risk_metrics.values()) / max(len(self.position_history), 1)
            }
        }
    
    def reset(self):
        """리스크 관리자 상태 리셋"""
        self.previous_weights = None
        self.position_history.clear()
        self.turnover_history.clear()
        self.risk_metrics = {
            'position_violations': 0,
            'turnover_violations': 0,
            'concentration_violations': 0
        }


class Calculations:
    """
    수치 계산을 위한 유틸리티 클래스
    """
    
    @staticmethod
    def divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """나눗셈"""
        if abs(denominator) < 1e-10:
            return default
        return numerator / denominator
    
    @staticmethod
    def log(value: float, default: float = 0.0) -> float:
        """로그 계산"""
        if value <= 0:
            return default
        return np.log(value)
    
    @staticmethod
    def sqrt(value: float, default: float = 0.0) -> float:
        """제곱근 계산"""
        if value < 0:
            return default
        return np.sqrt(value)
    
    @staticmethod
    def normalize(array: np.ndarray, default_value: float = 0.0) -> np.ndarray:
        """정규화"""
        if np.sum(array) == 0:
            return np.full_like(array, default_value)
        return array / np.sum(array)
    
    @staticmethod
    def mean(x):
        """평균 계산"""
        if len(x) == 0 or x.isnull().all():
            return 0.0
        return x.mean() if not np.isnan(x.mean()) else 0.0
    
    @staticmethod
    def std(x):
        """표준편차 계산"""
        if len(x) == 0 or x.isnull().all():
            return 0.0
        return x.std() if not np.isnan(x.std()) else 0.0
    
    @staticmethod
    def corr(x):
        """상관계수 계산"""
        try:
            if len(x) <= 1 or x.isnull().all().all():
                return 0.0
            corr_matrix = np.corrcoef(x.T)
            if np.isnan(corr_matrix).any():
                return 0.0
            return np.mean(corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)])
        except:
            return 0.0
    
    @staticmethod
    def momentum(x):
        """모멘텀 계산"""
        try:
            if len(x) < 5:
                return 0.0
            momentum = x.iloc[-1] / x.iloc[-5] - 1
            return momentum.mean() if not np.isnan(momentum.mean()) else 0.0
        except:
            return 0.0
    
    @staticmethod
    def skew(x):
        """왜도 계산"""
        try:
            skew_vals = x.skew()
            if skew_vals.isnull().all():
                return 0.0
            return skew_vals.mean() if not np.isnan(skew_vals.mean()) else 0.0
        except:
            return 0.0
    
    @staticmethod
    def kurtosis(x):
        """첨도 계산"""
        try:
            kurt_vals = x.kurtosis()
            if kurt_vals.isnull().all():
                return 0.0
            return kurt_vals.mean() if not np.isnan(kurt_vals.mean()) else 0.0
        except:
            return 0.0
    
    @staticmethod
    def trend(x):
        """트렌드 계산"""
        try:
            if len(x) < 2:
                return 0.0
            # 단순 선형 회귀 기울기
            n = len(x)
            x_vals = np.arange(n)
            slope = (n * np.sum(x_vals * x) - np.sum(x_vals) * np.sum(x)) / (n * np.sum(x_vals**2) - np.sum(x_vals)**2)
            return slope if not np.isnan(slope) else 0.0
        except:
            return 0.0
    
    @staticmethod
    def correlation(x: np.ndarray, y: np.ndarray, default: float = 0.0) -> float:
        """상관계수 계산"""
        if len(x) != len(y) or len(x) < 2:
            return default
        
        try:
            corr = np.corrcoef(x, y)[0, 1]
            return corr if not np.isnan(corr) else default
        except:
            return default
    
    @staticmethod
    def volatility(returns: np.ndarray, default: float = 0.02) -> float:
        """변동성 계산"""
        if len(returns) < 2:
            return default
        
        try:
            vol = np.std(returns)
            return vol if not np.isnan(vol) else default
        except:
            return default
    
    @staticmethod
    def clip_weights(weights: np.ndarray, 
                    min_weight: float = 0.0, 
                    max_weight: float = 1.0) -> np.ndarray:
        """가중치 클리핑"""
        return np.clip(weights, min_weight, max_weight)
    
    @staticmethod
    def remove_outliers(data: np.ndarray, 
                       z_threshold: float = 3.0) -> np.ndarray:
        """이상값 제거"""
        if len(data) < 3:
            return data
        
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return data[z_scores < z_threshold]
    
    @staticmethod
    def smooth_weights(current_weights: np.ndarray,
                      previous_weights: np.ndarray,
                      smoothing_factor: float = 0.8) -> np.ndarray:
        """가중치 스무딩"""
        if previous_weights is None:
            return current_weights
        
        return (smoothing_factor * current_weights + 
                (1 - smoothing_factor) * previous_weights)