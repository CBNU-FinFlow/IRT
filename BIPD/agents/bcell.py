"""
B-Cell 전략 생성 시스템
생체면역 시스템의 B-Cell을 모델링한 전문화된 투자 전략 생성 컴포넌트
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from typing import Dict, List, Optional, Any
from datetime import datetime
import random


class StrategyNetwork(nn.Module):
    """B-Cell 전략 생성 신경망"""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64):
        super(StrategyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class BCell:
    """
    B-Cell 전문화된 투자 전략 시스템
    특정 시장 상황에 특화된 포트폴리오 전략 생성
    """
    
    def __init__(self, 
                 cell_id: str,
                 specialty_type: str,
                 n_assets: int,
                 input_size: int = 12):
        self.cell_id = cell_id
        self.specialty_type = specialty_type
        self.n_assets = n_assets
        self.input_size = input_size
        
        # 신경망 초기화
        self.strategy_network = StrategyNetwork(input_size, n_assets)
        self.optimizer = optim.Adam(self.strategy_network.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=50
        )
        
        # 경험 버퍼
        self.experience_buffer = deque(maxlen=1000)
        self.specialist_buffer = deque(maxlen=500)
        
        # 전문화 설정
        self.specialization_strength = 0.1
        self.activation_threshold = self._get_activation_threshold()
        self.activation_level = 0.0
        
        # 학습 상태
        self.training_mode = True
        self.update_count = 0
        
        # 성능 추적
        self.performance_history = deque(maxlen=100)
        self.decision_history = deque(maxlen=1000)
        
        # 전문화 조건
        self.specialty_conditions = self._get_specialty_conditions()
    
    def _get_activation_threshold(self) -> float:
        """전문화 유형별 활성화 임계값"""
        thresholds = {
            'volatility': 0.6,
            'correlation': 0.3,
            'momentum': 0.5,
            'liquidity': 0.4,
            'memory_recall': 0.5
        }
        return thresholds.get(self.specialty_type, 0.5)
    
    def _get_specialty_conditions(self) -> Dict[str, Any]:
        """전문화 조건 설정"""
        conditions = {
            'volatility': {
                'feature_index': 1,
                'threshold': 0.6,
                'description': "시장 변동성 위험에 특화된 안전 자산 중심 포트폴리오 구성"
            },
            'correlation': {
                'feature_index': 2,
                'threshold': 0.3,
                'center': 0.5,
                'description': "상관관계 위험에 특화된 분산 투자 전략 적용"
            },
            'momentum': {
                'feature_index': 3,
                'threshold': 0.5,
                'description': "모멘텀 위험에 특화된 추세 추종 전략 활용"
            },
            'liquidity': {
                'feature_index': 4,
                'threshold': 0.4,
                'inverse': True,
                'description': "유동성 위험에 특화된 대형주 중심 포트폴리오 구성"
            },
            'memory_recall': {
                'feature_index': None,
                'threshold': 0.5,
                'description': "과거 위기 경험을 바탕으로 한 검증된 대응 전략 적용"
            }
        }
        return conditions.get(self.specialty_type, {})
    
    def is_specialty_situation(self, market_features: np.ndarray, crisis_level: float) -> bool:
        """전문화 상황 판단"""
        if not self.specialty_conditions:
            return False
        
        # memory_recall 전문화는 위기 수준 기반
        if self.specialty_type == 'memory_recall':
            return crisis_level > self.specialty_conditions['threshold']
        
        # 기타 전문화는 특성 기반
        feature_idx = self.specialty_conditions.get('feature_index')
        if feature_idx is None or feature_idx >= len(market_features):
            return False
        
        feature_value = market_features[feature_idx]
        threshold = self.specialty_conditions['threshold']
        
        # 조건 확인
        if 'center' in self.specialty_conditions:
            center = self.specialty_conditions['center']
            return abs(feature_value - center) > threshold
        elif self.specialty_conditions.get('inverse', False):
            return feature_value < threshold
        else:
            return feature_value > threshold
    
    def calculate_activation_strength(self, market_features: np.ndarray, crisis_level: float) -> float:
        """활성화 강도 계산"""
        if not self.specialty_conditions:
            return 0.0
        
        # memory_recall 전문화
        if self.specialty_type == 'memory_recall':
            threshold = self.specialty_conditions['threshold']
            return max(0, (crisis_level - threshold) / threshold)
        
        # 기타 전문화
        feature_idx = self.specialty_conditions.get('feature_index')
        if feature_idx is None or feature_idx >= len(market_features):
            return 0.0
        
        feature_value = market_features[feature_idx]
        threshold = self.specialty_conditions['threshold']
        
        if 'center' in self.specialty_conditions:
            center = self.specialty_conditions['center']
            deviation = abs(feature_value - center)
            strength = max(0, (deviation - threshold) / threshold)
        elif self.specialty_conditions.get('inverse', False):
            strength = max(0, (threshold - feature_value) / threshold)
        else:
            strength = max(0, (feature_value - threshold) / threshold)
        
        return min(strength, 1.0)
    
    def generate_portfolio_weights(self, market_features: np.ndarray, crisis_level: float) -> np.ndarray:
        """포트폴리오 가중치 생성"""
        # 활성화 강도 계산
        activation_strength = self.calculate_activation_strength(market_features, crisis_level)
        self.activation_level = activation_strength
        
        # 신경망 입력 준비
        if len(market_features) < self.input_size:
            # 패딩
            padded_features = np.zeros(self.input_size)
            padded_features[:len(market_features)] = market_features
            features = padded_features
        else:
            features = market_features[:self.input_size]
        
        # 신경망 추론
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(features).unsqueeze(0)
            weights = self.strategy_network(feature_tensor).squeeze().numpy()
        
        # 전문화 조정
        if self.is_specialty_situation(market_features, crisis_level):
            weights = self._apply_specialization(weights, activation_strength)
        
        return weights
    
    def _apply_specialization(self, weights: np.ndarray, activation_strength: float) -> np.ndarray:
        """전문화 조정 적용"""
        adjusted_weights = weights.copy()
        
        # 전문화 유형별 조정
        if self.specialty_type == 'volatility':
            # 안전 자산 선호 (첫 번째 자산을 안전 자산으로 가정)
            adjusted_weights[0] += activation_strength * 0.3
        elif self.specialty_type == 'correlation':
            # 분산 투자 강화
            uniform_weight = 1.0 / self.n_assets
            adjusted_weights = (1 - activation_strength) * adjusted_weights + activation_strength * uniform_weight
        elif self.specialty_type == 'momentum':
            # 모멘텀 강화 (상위 자산 집중)
            top_indices = np.argsort(weights)[-2:]
            boost = activation_strength * 0.2
            adjusted_weights[top_indices] += boost / len(top_indices)
        elif self.specialty_type == 'liquidity':
            # 대형주 선호 (처음 3개 자산을 대형주로 가정)
            large_cap_boost = activation_strength * 0.25
            adjusted_weights[:3] += large_cap_boost / 3
        elif self.specialty_type == 'memory_recall':
            # 과거 성공 패턴 재현 (보수적 접근)
            adjusted_weights *= (1 - activation_strength * 0.1)
        
        # 정규화
        adjusted_weights = np.maximum(adjusted_weights, 0)
        weight_sum = np.sum(adjusted_weights)
        if weight_sum > 0:
            adjusted_weights = adjusted_weights / weight_sum
        else:
            adjusted_weights = np.ones(self.n_assets) / self.n_assets
        
        return adjusted_weights
    
    def store_experience(self, market_features: np.ndarray, action: np.ndarray, 
                        reward: float, next_features: np.ndarray, crisis_level: float):
        """경험 저장"""
        experience = {
            'state': market_features.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_state': next_features.copy(),
            'crisis_level': crisis_level,
            'timestamp': datetime.now().isoformat()
        }
        
        self.experience_buffer.append(experience)
        
        # 전문화 상황이면 전문 버퍼에도 저장
        if self.is_specialty_situation(market_features, crisis_level):
            self.specialist_buffer.append(experience)
    
    def learn(self, batch_size: int = 32):
        """배치 학습"""
        if not self.training_mode:
            return
        
        # 전문 버퍼 우선 사용
        if len(self.specialist_buffer) >= batch_size:
            batch = random.sample(list(self.specialist_buffer), batch_size)
        elif len(self.experience_buffer) >= batch_size:
            batch = random.sample(list(self.experience_buffer), batch_size)
        else:
            return
        
        # 배치 데이터 준비
        states = torch.FloatTensor([exp['state'][:self.input_size] for exp in batch])
        actions = torch.FloatTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        
        # 손실 계산
        predicted_actions = self.strategy_network(states)
        
        # 정책 그래디언트 손실
        log_probs = torch.log(predicted_actions + 1e-8)
        action_probs = torch.sum(log_probs * actions, dim=1)
        loss = -torch.mean(action_probs * rewards)
        
        # 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 성능 추적
        avg_reward = torch.mean(rewards).item()
        self.performance_history.append(avg_reward)
        self.scheduler.step(avg_reward)
        
        self.update_count += 1
        
        # 전문화 강도 업데이트
        if avg_reward > 0:
            self.specialization_strength = min(self.specialization_strength * 1.01, 1.0)
            
    def train_step(self, market_features, target_action):
        """
        단일 훈련 스텝 (사전 훈련용)
        
        Args:
            market_features: 시장 특성 벡터
            target_action: 타겟 액션 (전문 정책으로부터)
        """
        # 입력 차원 맞추기
        if len(market_features) > self.input_size:
            market_features = market_features[:self.input_size]
        elif len(market_features) < self.input_size:
            # 패딩
            padded_features = np.zeros(self.input_size)
            padded_features[:len(market_features)] = market_features
            market_features = padded_features
            
        # 타겟 액션 정규화
        target_action = np.abs(target_action)
        if np.sum(target_action) > 0:
            target_action = target_action / np.sum(target_action)
        else:
            target_action = np.ones(self.n_assets) / self.n_assets
            
        # 텐서 변환
        state_tensor = torch.FloatTensor(market_features).unsqueeze(0)
        target_tensor = torch.FloatTensor(target_action).unsqueeze(0)
        
        # 순전파
        predicted_action = self.strategy_network(state_tensor)
        
        # 손실 계산 (MSE)
        loss = F.mse_loss(predicted_action, target_tensor)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def generate_strategy(self, market_features: np.ndarray) -> np.ndarray:
        """
        전략 생성 (generate_portfolio_weights의 별칭)
        
        Args:
            market_features: 시장 특성 벡터
            
        Returns:
            포트폴리오 가중치
        """
        # 기본 위기 수준 0.0으로 설정
        crisis_level = 0.0
        return self.generate_portfolio_weights(market_features, crisis_level)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """성능 지표 반환"""
        if not self.performance_history:
            return {'avg_reward': 0.0, 'stability': 0.0, 'update_count': self.update_count}
        
        rewards = list(self.performance_history)
        return {
            'avg_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'stability': 1.0 - np.std(rewards) / (np.mean(rewards) + 1e-8),
            'update_count': self.update_count,
            'specialization_strength': self.specialization_strength
        }
    
    def get_decision_explanation(self, market_features: np.ndarray, crisis_level: float) -> Dict[str, Any]:
        """의사결정 설명 생성"""
        is_specialty = self.is_specialty_situation(market_features, crisis_level)
        activation_strength = self.calculate_activation_strength(market_features, crisis_level)
        
        explanation = {
            'cell_id': self.cell_id,
            'specialty_type': self.specialty_type,
            'is_specialty_situation': is_specialty,
            'activation_strength': activation_strength,
            'strategy_description': self.specialty_conditions.get('description', ''),
            'reasoning': self._generate_reasoning(market_features, crisis_level, is_specialty, activation_strength)
        }
        
        return explanation
    
    def _generate_reasoning(self, market_features: np.ndarray, crisis_level: float, 
                           is_specialty: bool, activation_strength: float) -> str:
        """의사결정 근거 생성"""
        if not is_specialty:
            return f"{self.specialty_type} 전문화 조건을 충족하지 않아 일반 전략 적용"
        
        reasoning_parts = []
        
        # 활성화 강도 기반 설명
        if activation_strength > 0.8:
            reasoning_parts.append(f"매우 강한 {self.specialty_type} 신호로 전문 전략 적극 적용")
        elif activation_strength > 0.6:
            reasoning_parts.append(f"강한 {self.specialty_type} 신호로 전문 전략 활발 적용")
        elif activation_strength > 0.4:
            reasoning_parts.append(f"중간 {self.specialty_type} 신호로 전문 전략 부분 적용")
        else:
            reasoning_parts.append(f"약한 {self.specialty_type} 신호로 전문 전략 제한 적용")
        
        # 위기 수준 기반 설명
        if crisis_level > 0.7:
            reasoning_parts.append("극심한 위기 상황에서 전문 대응 전략 활성화")
        elif crisis_level > 0.5:
            reasoning_parts.append("높은 위기 상황에서 전문 대응 전략 적용")
        elif crisis_level > 0.3:
            reasoning_parts.append("중간 위기 상황에서 전문 대응 전략 부분 적용")
        
        return "; ".join(reasoning_parts)
    
    def set_training_mode(self, training: bool):
        """훈련 모드 설정"""
        self.training_mode = training
        self.strategy_network.train() if training else self.strategy_network.eval()
    
    def reset(self):
        """B-Cell 상태 리셋"""
        self.activation_level = 0.0
        self.experience_buffer.clear()
        self.specialist_buffer.clear()
        self.performance_history.clear()
        self.decision_history.clear()
        self.update_count = 0
        self.specialization_strength = 0.1