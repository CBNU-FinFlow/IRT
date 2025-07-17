"""
BIPD (Behavioral Immune Portfolio Defense) Agent
생체면역 기반 강화학습 에이전트

논문 기여도:
1. 생체면역 시스템에서 영감받은 새로운 강화학습 아키텍처
2. 시계열 데이터 특화 메모리 메커니즘
3. 다중 전문가 정책 네트워크 (B-Cell Network)
4. 적응적 위기 감지 시스템 (T-Cell System)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import random


class CytokineNetwork(nn.Module):
    """
    Cytokine Network: 세포간 통신 및 협력 메커니즘
    - T-Cell과 B-Cell 간 정보 전달
    - 글로벌 상태 정보 공유
    - 협력적 의사결정 지원
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(CytokineNetwork, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, tcell_signals: torch.Tensor, bcell_signals: torch.Tensor) -> torch.Tensor:
        """
        T-Cell과 B-Cell 신호를 통합하여 협력적 의사결정 지원
        """
        # 신호 통합
        combined_signals = torch.cat([tcell_signals, bcell_signals], dim=1)
        
        # 주의 메커니즘을 통한 중요 정보 추출
        attn_output, _ = self.attention(combined_signals, combined_signals, combined_signals)
        
        # 통합된 신호 처리
        x = F.relu(self.fc1(attn_output.mean(dim=1)))
        x = F.relu(self.fc2(x))
        coordination_signal = self.output_layer(x)
        
        return coordination_signal


class AdaptiveTCell(nn.Module):
    """
    적응적 T-Cell: 위기 감지 및 상태 평가
    - 동적 임계값 조정
    - 시계열 패턴 학습
    - 메모리 기반 위기 예측
    """
    
    def __init__(self, input_dim: int, cell_id: str, sensitivity: float = 0.1):
        super(AdaptiveTCell, self).__init__()
        self.cell_id = cell_id
        self.base_sensitivity = sensitivity
        self.current_sensitivity = sensitivity
        
        # 위기 감지 네트워크
        self.crisis_detector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 적응적 임계값 조정 네트워크
        self.threshold_adapter = nn.Sequential(
            nn.Linear(input_dim + 1, 32),  # +1 for current crisis level
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 시계열 패턴 학습을 위한 LSTM
        self.temporal_lstm = nn.LSTM(input_dim, 32, batch_first=True)
        
        # 메모리 버퍼
        self.crisis_memory = deque(maxlen=1000)
        self.performance_history = deque(maxlen=100)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        
    def detect_crisis(self, market_features: torch.Tensor, 
                     sequence_data: Optional[torch.Tensor] = None) -> Dict:
        """
        위기 감지 및 상태 평가
        """
        # 기본 위기 수준 계산
        crisis_level = self.crisis_detector(market_features).item()
        
        # 시계열 패턴 분석 (LSTM 사용)
        temporal_signal = 0.0
        if sequence_data is not None:
            lstm_out, _ = self.temporal_lstm(sequence_data.unsqueeze(0))
            temporal_signal = lstm_out[:, -1, :].mean().item()
        
        # 메모리 기반 위기 예측
        memory_signal = self._recall_crisis_memory(market_features)
        
        # 통합 위기 수준 계산
        integrated_crisis_level = (
            0.5 * crisis_level + 
            0.3 * temporal_signal + 
            0.2 * memory_signal
        )
        
        # 적응적 임계값 조정
        threshold_input = torch.cat([market_features, torch.tensor([integrated_crisis_level])])
        adjusted_threshold = self.threshold_adapter(threshold_input).item()
        
        # 위기 감지 결과
        is_crisis = integrated_crisis_level > adjusted_threshold
        
        # 메모리 업데이트
        self.crisis_memory.append({
            'features': market_features.detach().numpy(),
            'crisis_level': integrated_crisis_level,
            'timestamp': len(self.crisis_memory)
        })
        
        return {
            'crisis_level': integrated_crisis_level,
            'threshold': adjusted_threshold,
            'is_crisis': is_crisis,
            'temporal_signal': temporal_signal,
            'memory_signal': memory_signal,
            'cell_id': self.cell_id
        }
    
    def _recall_crisis_memory(self, current_features: torch.Tensor) -> float:
        """
        메모리 기반 위기 예측
        """
        if len(self.crisis_memory) < 10:
            return 0.0
            
        # 현재 상황과 유사한 과거 상황 찾기
        current_np = current_features.detach().numpy()
        similarities = []
        
        for memory in list(self.crisis_memory)[-100:]:  # 최근 100개 메모리만 사용
            similarity = np.exp(-np.linalg.norm(current_np - memory['features']) / 10)
            similarities.append(similarity * memory['crisis_level'])
        
        if similarities:
            return np.mean(similarities)
        return 0.0
    
    def update_performance(self, reward: float):
        """
        성과 기반 민감도 조정
        """
        self.performance_history.append(reward)
        
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(list(self.performance_history)[-10:])
            
            # 성과가 좋으면 민감도 감소 (과민 반응 방지)
            # 성과가 나쁘면 민감도 증가 (위기 감지 강화)
            if recent_performance > 0.1:
                self.current_sensitivity = max(0.01, self.current_sensitivity * 0.95)
            elif recent_performance < -0.1:
                self.current_sensitivity = min(0.5, self.current_sensitivity * 1.05)


class SpecializedBCell(nn.Module):
    """
    전문화된 B-Cell: 특정 리스크에 특화된 정책 학습
    - 각기 다른 시장 상황에 대한 전문 전략
    - 협력적 정책 학습
    - 적응적 전문화 강화
    """
    
    def __init__(self, input_dim: int, n_assets: int, specialization: str, 
                 cell_id: str, hidden_dim: int = 64):
        super(SpecializedBCell, self).__init__()
        self.cell_id = cell_id
        self.specialization = specialization
        self.n_assets = n_assets
        
        # 전문화별 특성 가중치
        self.specialization_weights = nn.Parameter(
            torch.randn(input_dim) * 0.1, requires_grad=True
        )
        
        # 정책 네트워크 (액터)
        self.actor = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for crisis level
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_assets),
            nn.Softmax(dim=-1)
        )
        
        # 가치 네트워크 (크리틱)
        self.critic = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 경험 버퍼
        self.experience_buffer = deque(maxlen=10000)
        
        # 전문화 성과 추적
        self.specialization_performance = deque(maxlen=1000)
        
        # 옵티마이저
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
    def generate_policy(self, market_features: torch.Tensor, 
                       crisis_level: float, 
                       coordination_signal: Optional[torch.Tensor] = None) -> Dict:
        """
        전문화된 정책 생성
        """
        # 전문화 가중치 적용
        specialized_features = market_features * self.specialization_weights
        
        # 협력 신호 통합
        if coordination_signal is not None:
            specialized_features = specialized_features + 0.1 * coordination_signal
            
        # 입력 준비
        policy_input = torch.cat([specialized_features, torch.tensor([crisis_level])])
        
        # 정책 생성 (액터)
        action_probs = self.actor(policy_input)
        
        # 가치 추정 (크리틱)
        state_value = self.critic(policy_input)
        
        # 행동 선택 (확률적 또는 탐욕적)
        action = torch.multinomial(action_probs, 1).item()
        
        return {
            'action_probs': action_probs,
            'action': action,
            'state_value': state_value,
            'confidence': action_probs.max().item(),
            'cell_id': self.cell_id,
            'specialization': self.specialization
        }
    
    def update_policy(self, reward: float, next_state_value: float, 
                     gamma: float = 0.99) -> Dict:
        """
        정책 업데이트 (Actor-Critic 방식)
        """
        if len(self.experience_buffer) < 1:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
            
        # 최근 경험 가져오기
        recent_experience = self.experience_buffer[-1]
        
        # TD 에러 계산
        td_target = reward + gamma * next_state_value
        td_error = td_target - recent_experience['state_value']
        
        # 크리틱 손실
        critic_loss = F.mse_loss(recent_experience['state_value'], td_target.detach())
        
        # 액터 손실 (정책 그래디언트)
        action_prob = recent_experience['action_probs'][recent_experience['action']]
        actor_loss = -torch.log(action_prob) * td_error.detach()
        
        # 옵티마이저 업데이트
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 전문화 성과 업데이트
        self.specialization_performance.append(reward)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'td_error': td_error.item()
        }
    
    def store_experience(self, state: torch.Tensor, action: int, 
                        action_probs: torch.Tensor, state_value: torch.Tensor,
                        reward: float):
        """
        경험 저장
        """
        experience = {
            'state': state,
            'action': action,
            'action_probs': action_probs,
            'state_value': state_value,
            'reward': reward,
            'timestamp': len(self.experience_buffer)
        }
        self.experience_buffer.append(experience)


class ImmuneMemoryCell:
    """
    면역 메모리 셀: 장기 기억 및 경험 저장
    - 시계열 특화 메모리 구조
    - 적응적 기억 강화/약화
    - 상황별 메모리 검색
    """
    
    def __init__(self, memory_size: int = 50000):
        self.memory_size = memory_size
        self.episodic_memory = deque(maxlen=memory_size)  # 에피소드 기억
        self.semantic_memory = {}  # 의미 기억 (패턴별)
        self.working_memory = deque(maxlen=100)  # 작업 기억
        
        # 메모리 강도 추적
        self.memory_strengths = deque(maxlen=memory_size)
        
    def store_episode(self, state: np.ndarray, action: int, reward: float, 
                     next_state: np.ndarray, done: bool, 
                     market_context: Dict):
        """
        에피소드 기억 저장
        """
        episode = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'market_context': market_context,
            'timestamp': len(self.episodic_memory),
            'access_count': 0
        }
        
        self.episodic_memory.append(episode)
        self.memory_strengths.append(abs(reward))  # 보상 크기에 따른 기억 강도
        
        # 의미 기억 업데이트
        self._update_semantic_memory(episode)
        
    def _update_semantic_memory(self, episode: Dict):
        """
        의미 기억 업데이트 (패턴 학습)
        """
        # 시장 상황 패턴 분류
        market_context = episode['market_context']
        volatility_level = 'high' if market_context.get('volatility', 0) > 0.02 else 'low'
        trend_direction = 'up' if market_context.get('trend', 0) > 0 else 'down'
        
        pattern_key = f"{volatility_level}_{trend_direction}"
        
        if pattern_key not in self.semantic_memory:
            self.semantic_memory[pattern_key] = {
                'episodes': [],
                'avg_reward': 0.0,
                'best_actions': {},
                'access_count': 0
            }
        
        # 패턴별 기억 업데이트
        pattern_memory = self.semantic_memory[pattern_key]
        pattern_memory['episodes'].append(episode)
        pattern_memory['avg_reward'] = np.mean([ep['reward'] for ep in pattern_memory['episodes']])
        
        # 최적 행동 업데이트
        action = episode['action']
        if action not in pattern_memory['best_actions']:
            pattern_memory['best_actions'][action] = []
        pattern_memory['best_actions'][action].append(episode['reward'])
        
    def recall_similar_episodes(self, current_state: np.ndarray, 
                               k: int = 10) -> List[Dict]:
        """
        유사 에피소드 회상
        """
        if len(self.episodic_memory) < k:
            return list(self.episodic_memory)
            
        # 유사도 계산
        similarities = []
        for i, episode in enumerate(self.episodic_memory):
            similarity = np.exp(-np.linalg.norm(current_state - episode['state']) / 10)
            # 메모리 강도와 접근 빈도 고려
            memory_strength = self.memory_strengths[i] if i < len(self.memory_strengths) else 1.0
            adjusted_similarity = similarity * memory_strength * (1 + episode['access_count'] * 0.1)
            similarities.append((adjusted_similarity, i, episode))
        
        # 상위 k개 선택
        similarities.sort(key=lambda x: x[0], reverse=True)
        selected_episodes = [ep for _, _, ep in similarities[:k]]
        
        # 접근 횟수 업데이트
        for _, idx, episode in similarities[:k]:
            episode['access_count'] += 1
            
        return selected_episodes
    
    def get_pattern_advice(self, market_context: Dict) -> Optional[Dict]:
        """
        패턴 기반 조언 제공
        """
        volatility_level = 'high' if market_context.get('volatility', 0) > 0.02 else 'low'
        trend_direction = 'up' if market_context.get('trend', 0) > 0 else 'down'
        pattern_key = f"{volatility_level}_{trend_direction}"
        
        if pattern_key in self.semantic_memory:
            pattern_memory = self.semantic_memory[pattern_key]
            pattern_memory['access_count'] += 1
            
            # 최적 행동 추천
            best_action = max(pattern_memory['best_actions'].items(), 
                            key=lambda x: np.mean(x[1]))[0]
            
            return {
                'recommended_action': best_action,
                'expected_reward': pattern_memory['avg_reward'],
                'confidence': len(pattern_memory['episodes']) / 100.0,
                'pattern': pattern_key
            }
        
        return None


class BIPDAgent:
    """
    BIPD (Behavioral Immune Portfolio Defense) 에이전트
    생체면역 시스템에서 영감받은 강화학습 에이전트
    """
    
    def __init__(self, input_dim: int, n_assets: int, n_tcells: int = 3, 
                 n_bcells: int = 5, memory_size: int = 50000):
        self.input_dim = input_dim
        self.n_assets = n_assets
        
        # T-Cell 네트워크 (위기 감지)
        self.tcells = [
            AdaptiveTCell(input_dim, f"T{i}", sensitivity=0.05 + i * 0.02)
            for i in range(n_tcells)
        ]
        
        # B-Cell 네트워크 (전문화된 정책)
        specializations = ['volatility', 'momentum', 'trend', 'correlation', 'liquidity']
        self.bcells = [
            SpecializedBCell(input_dim, n_assets, specializations[i % len(specializations)], f"B{i}")
            for i in range(n_bcells)
        ]
        
        # Cytokine 네트워크 (세포간 통신)
        self.cytokine_network = CytokineNetwork(input_dim)
        
        # 면역 메모리 셀
        self.memory_cell = ImmuneMemoryCell(memory_size)
        
        # 시스템 상태
        self.system_state = {
            'crisis_level': 0.0,
            'active_bcells': [],
            'memory_strength': 0.0,
            'coordination_strength': 0.0
        }
        
        # 성과 추적
        self.performance_history = deque(maxlen=1000)
        
    def select_action(self, market_features: torch.Tensor, 
                     sequence_data: Optional[torch.Tensor] = None,
                     market_context: Optional[Dict] = None) -> Dict:
        """
        행동 선택 (면역 반응)
        """
        # 1. T-Cell 위기 감지
        tcell_responses = []
        for tcell in self.tcells:
            response = tcell.detect_crisis(market_features, sequence_data)
            tcell_responses.append(response)
        
        # 통합 위기 수준 계산
        crisis_level = np.mean([r['crisis_level'] for r in tcell_responses])
        self.system_state['crisis_level'] = crisis_level
        
        # 2. 메모리 검색
        memory_advice = None
        if market_context:
            memory_advice = self.memory_cell.get_pattern_advice(market_context)
        
        # 3. B-Cell 정책 생성
        bcell_responses = []
        bcell_signals = []
        
        for bcell in self.bcells:
            response = bcell.generate_policy(market_features, crisis_level)
            bcell_responses.append(response)
            bcell_signals.append(response['action_probs'])
        
        # 4. Cytokine 네트워크를 통한 협력
        tcell_signals = torch.stack([torch.tensor([r['crisis_level']]) for r in tcell_responses])
        bcell_signals_tensor = torch.stack(bcell_signals)
        
        coordination_signal = self.cytokine_network(
            tcell_signals.unsqueeze(0), 
            bcell_signals_tensor.unsqueeze(0)
        ).squeeze(0)
        
        # 5. 최종 행동 선택 (가중 투표)
        final_action_probs = torch.zeros(self.n_assets)
        total_confidence = 0.0
        
        for i, response in enumerate(bcell_responses):
            confidence = response['confidence']
            # 위기 상황에서는 더 보수적인 B-Cell 선호
            if crisis_level > 0.7 and response['specialization'] in ['volatility', 'liquidity']:
                confidence *= 1.5
            
            final_action_probs += confidence * response['action_probs']
            total_confidence += confidence
        
        if total_confidence > 0:
            final_action_probs /= total_confidence
        
        # 메모리 조언 통합
        if memory_advice and memory_advice['confidence'] > 0.5:
            memory_weight = memory_advice['confidence'] * 0.3
            final_action_probs = (1 - memory_weight) * final_action_probs
            final_action_probs[memory_advice['recommended_action']] += memory_weight
        
        # 최종 행동 선택
        action = torch.multinomial(final_action_probs, 1).item()
        
        return {
            'action': action,
            'action_probs': final_action_probs,
            'crisis_level': crisis_level,
            'tcell_responses': tcell_responses,
            'bcell_responses': bcell_responses,
            'coordination_signal': coordination_signal,
            'memory_advice': memory_advice
        }
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool, market_context: Dict) -> Dict:
        """
        에이전트 업데이트
        """
        # 1. 메모리 저장
        self.memory_cell.store_episode(
            state, action, reward, next_state, done, market_context
        )
        
        # 2. T-Cell 성과 업데이트
        for tcell in self.tcells:
            tcell.update_performance(reward)
        
        # 3. B-Cell 정책 업데이트
        update_results = {}
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        
        for bcell in self.bcells:
            # 다음 상태 가치 추정
            next_state_value = bcell.critic(
                torch.cat([next_state_tensor, torch.tensor([self.system_state['crisis_level']])])
            )
            
            # 정책 업데이트
            result = bcell.update_policy(reward, next_state_value.item())
            update_results[bcell.cell_id] = result
        
        # 4. 성과 추적
        self.performance_history.append(reward)
        
        return {
            'update_results': update_results,
            'avg_performance': np.mean(list(self.performance_history)[-100:]),
            'system_state': self.system_state.copy()
        }
    
    def get_system_diagnostics(self) -> Dict:
        """
        시스템 진단 정보
        """
        diagnostics = {
            'tcell_diagnostics': [
                {
                    'cell_id': tcell.cell_id,
                    'sensitivity': tcell.current_sensitivity,
                    'memory_size': len(tcell.crisis_memory),
                    'avg_performance': np.mean(list(tcell.performance_history)[-10:]) if tcell.performance_history else 0.0
                }
                for tcell in self.tcells
            ],
            'bcell_diagnostics': [
                {
                    'cell_id': bcell.cell_id,
                    'specialization': bcell.specialization,
                    'experience_count': len(bcell.experience_buffer),
                    'avg_specialization_performance': np.mean(list(bcell.specialization_performance)[-10:]) if bcell.specialization_performance else 0.0
                }
                for bcell in self.bcells
            ],
            'memory_diagnostics': {
                'episodic_memory_size': len(self.memory_cell.episodic_memory),
                'semantic_patterns': len(self.memory_cell.semantic_memory),
                'working_memory_size': len(self.memory_cell.working_memory)
            },
            'system_state': self.system_state,
            'overall_performance': np.mean(list(self.performance_history)[-100:]) if self.performance_history else 0.0
        }
        
        return diagnostics