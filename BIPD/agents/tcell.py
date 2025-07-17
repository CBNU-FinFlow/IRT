"""
T-Cell 위기 감지 시스템
생체면역 시스템의 T-Cell을 모델링한 위기 감지 및 설명가능성 컴포넌트
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque


class TCell:
    """
    T-Cell 위기 감지 시스템
    시장 이상 징후를 감지하고 위기 수준을 평가
    """
    
    def __init__(self, cell_id: str, sensitivity: float = 0.05):
        self.cell_id = cell_id
        self.sensitivity = sensitivity
        self.activation_threshold = 0.15
        self.activation_level = 0.0
        self.memory_strength = 0.5
        
        # 이상 감지 모델
        self.anomaly_detector = IsolationForest(
            contamination=sensitivity,
            random_state=42,
            n_estimators=100
        )
        
        # 히스토리 저장
        self.score_history = deque(maxlen=1000)
        self.market_state_history = deque(maxlen=1000)
        self.crisis_log = []
        
        # 위기 분류 임계값
        self.crisis_thresholds = {
            'severe': 0.7,
            'high': 0.5,
            'moderate': 0.3,
            'mild': 0.15
        }
        
        # 특성 가중치
        self.feature_weights = {
            'anomaly_extreme': 0.8,
            'anomaly_high': 0.6,
            'anomaly_moderate': 0.4,
            'anomaly_mild': 0.2,
            'volatility_boost': 0.2,
            'stress_boost': 0.15,
            'correlation_boost': 0.1,
            'trend_boost': 0.1
        }
        
        self.is_fitted = False
    
    def fit(self, market_data: np.ndarray):
        """이상 감지 모델 훈련"""
        if len(market_data) < 10:
            return
            
        self.anomaly_detector.fit(market_data)
        self.is_fitted = True
    
    def detect_crisis(self, market_features: np.ndarray) -> Dict[str, Any]:
        """
        위기 감지 및 분석
        
        Args:
            market_features: 시장 특성 벡터
            
        Returns:
            위기 감지 결과
        """
        if not self.is_fitted:
            # 훈련되지 않은 경우 기본 위기 감지
            return self._basic_crisis_detection(market_features)
        
        # 이상 점수 계산
        anomaly_score = self.anomaly_detector.decision_function([market_features])[0]
        
        # Z-score 계산
        z_score = self._calculate_z_score(anomaly_score)
        
        # 위기 기여도 분석
        crisis_contributions = self._analyze_crisis_contributions(z_score, market_features)
        
        # 위기 수준 계산
        crisis_level = self._calculate_crisis_level(crisis_contributions)
        
        # 위기 분류
        crisis_classification = self._classify_crisis_level(crisis_level)
        
        # 활성화 수준 업데이트
        self.activation_level = crisis_level
        
        # 의사결정 근거 생성
        decision_reasoning = self._generate_decision_reasoning(
            z_score, crisis_contributions, crisis_level, crisis_classification
        )
        
        # 결과 생성
        result = {
            'cell_id': self.cell_id,
            'timestamp': datetime.now().isoformat(),
            'anomaly_score': anomaly_score,
            'z_score': z_score,
            'crisis_level': crisis_level,
            'crisis_classification': crisis_classification,
            'activation_level': self.activation_level,
            'crisis_contributions': crisis_contributions,
            'decision_reasoning': decision_reasoning,
            'is_activated': crisis_level > self.activation_threshold
        }
        
        # 히스토리 업데이트
        self.score_history.append(anomaly_score)
        self.crisis_log.append(result)
        
        return result
    
    def _basic_crisis_detection(self, market_features: np.ndarray) -> Dict[str, Any]:
        """기본 위기 감지 (모델 훈련 전)"""
        # 변동성 기반 단순 위기 감지
        volatility = market_features[1] if len(market_features) > 1 else 0.02
        crisis_level = min(volatility / 0.05, 1.0)
        
        return {
            'cell_id': self.cell_id,
            'timestamp': datetime.now().isoformat(),
            'anomaly_score': 0.0,
            'z_score': 0.0,
            'crisis_level': crisis_level,
            'crisis_classification': self._classify_crisis_level(crisis_level),
            'activation_level': crisis_level,
            'crisis_contributions': {'volatility_boost': crisis_level * 0.2},
            'decision_reasoning': f"기본 변동성 기반 위기 감지 (수준: {crisis_level:.3f})",
            'is_activated': crisis_level > self.activation_threshold
        }
    
    def _calculate_z_score(self, current_score: float) -> float:
        """Z-score 계산"""
        if len(self.score_history) < 2:
            return 0.0
            
        scores = list(self.score_history)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if std_score == 0:
            return 0.0
            
        return (current_score - mean_score) / std_score
    
    def _analyze_crisis_contributions(self, z_score: float, market_features: np.ndarray) -> Dict[str, float]:
        """위기 기여도 분석"""
        contributions = {}
        
        # 이상 점수 기여도
        if z_score < -1.5:
            contributions['anomaly_extreme'] = self.feature_weights['anomaly_extreme']
        elif z_score < -1.0:
            contributions['anomaly_high'] = self.feature_weights['anomaly_high']
        elif z_score < -0.5:
            contributions['anomaly_moderate'] = self.feature_weights['anomaly_moderate']
        elif z_score < 0.0:
            contributions['anomaly_mild'] = self.feature_weights['anomaly_mild']
        
        # 시장 특성 기여도
        if len(market_features) > 1:
            volatility = market_features[1]
            if volatility > 0.3:
                contributions['volatility_boost'] = self.feature_weights['volatility_boost']
        
        if len(market_features) > 5:
            stress = market_features[5]
            if stress > 0.5:
                contributions['stress_boost'] = self.feature_weights['stress_boost']
        
        if len(market_features) > 2:
            correlation = market_features[2]
            if correlation > 0.8:
                contributions['correlation_boost'] = self.feature_weights['correlation_boost']
        
        return contributions
    
    def _calculate_crisis_level(self, contributions: Dict[str, float]) -> float:
        """위기 수준 계산"""
        return sum(contributions.values())
    
    def _classify_crisis_level(self, crisis_level: float) -> str:
        """위기 수준 분류"""
        if crisis_level > self.crisis_thresholds['severe']:
            return 'severe'
        elif crisis_level > self.crisis_thresholds['high']:
            return 'high'
        elif crisis_level > self.crisis_thresholds['moderate']:
            return 'moderate'
        elif crisis_level > self.crisis_thresholds['mild']:
            return 'mild'
        else:
            return 'normal'
    
    def _generate_decision_reasoning(self, 
                                   z_score: float,
                                   contributions: Dict[str, float],
                                   crisis_level: float,
                                   crisis_classification: str) -> str:
        """의사결정 근거 생성"""
        reasoning_parts = []
        
        # Z-score 기반 논리
        if z_score < -1.5:
            reasoning_parts.append("이상 점수가 역사적 평균 대비 1.5 표준편차 이상 하락하여 극심한 위기 신호")
        elif z_score < -1.0:
            reasoning_parts.append("이상 점수가 역사적 평균 대비 1.0 표준편차 이상 하락하여 높은 위기 신호")
        elif z_score < -0.5:
            reasoning_parts.append("이상 점수가 역사적 평균 대비 0.5 표준편차 이상 하락하여 중간 위기 신호")
        
        # 위기 수준 기반 논리
        if crisis_classification == 'severe':
            reasoning_parts.append("위기 수준이 심각 단계에 도달하여 최대 방어 모드 활성화")
        elif crisis_classification == 'high':
            reasoning_parts.append("위기 수준이 높은 단계에 도달하여 적극적 리스크 관리 필요")
        elif crisis_classification == 'moderate':
            reasoning_parts.append("위기 수준이 중간 단계에 도달하여 균형잡힌 대응 전략 적용")
        elif crisis_classification == 'mild':
            reasoning_parts.append("위기 수준이 경미한 단계에 도달하여 선택적 리스크 대응")
        
        # 주요 기여 요인 논리
        if contributions:
            max_contribution = max(contributions.values())
            if max_contribution > 0.5:
                reasoning_parts.append("주요 위기 요인이 임계치를 크게 초과하여 즉시 대응 필요")
            elif max_contribution > 0.3:
                reasoning_parts.append("주요 위기 요인이 임계치를 초과하여 적극적 대응 필요")
        
        return "; ".join(reasoning_parts) if reasoning_parts else "정상 상태로 특별한 대응 불필요"
    
    def get_crisis_history(self) -> List[Dict[str, Any]]:
        """위기 감지 히스토리 반환"""
        return list(self.crisis_log)
    
    def reset(self):
        """T-Cell 상태 리셋"""
        self.activation_level = 0.0
        self.score_history.clear()
        self.market_state_history.clear()
        self.crisis_log.clear()
        self.is_fitted = False