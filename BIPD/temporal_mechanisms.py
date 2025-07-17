"""
시계열 특화 메커니즘
생체면역 시스템의 시간적 특성을 모델링한 시계열 특화 컴포넌트

논문 기여도:
1. 면역 기억의 시간적 특성 모델링
2. 적응적 시간 윈도우 조정
3. 순환 패턴 및 계절성 감지
4. 시계열 특화 상태 표현
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    StandardScaler = None
    PCA = None
import warnings
warnings.filterwarnings('ignore')


class AdaptiveTimeWindowManager:
    """
    적응적 시간 윈도우 관리
    - 시장 상황에 따른 동적 윈도우 조정
    - 변동성 기반 적응적 lookback
    - 계절성 및 순환 패턴 고려
    """
    
    def __init__(self, 
                 base_window: int = 20,
                 min_window: int = 5,
                 max_window: int = 100,
                 volatility_threshold: float = 0.02):
        self.base_window = base_window
        self.min_window = min_window
        self.max_window = max_window
        self.volatility_threshold = volatility_threshold
        
        # 적응적 윈도우 히스토리
        self.window_history = deque(maxlen=252)
        self.volatility_history = deque(maxlen=252)
        
        # 계절성 패턴 추적
        self.seasonal_patterns = {}
        self.cycle_detector = CyclicalPatternDetector()
        
    def calculate_adaptive_window(self, 
                                market_data: pd.DataFrame,
                                current_volatility: float,
                                crisis_level: float = 0.0) -> int:
        """
        적응적 시간 윈도우 계산
        """
        # 1. 변동성 기반 조정
        volatility_factor = self._calculate_volatility_factor(current_volatility)
        
        # 2. 위기 수준 기반 조정
        crisis_factor = self._calculate_crisis_factor(crisis_level)
        
        # 3. 계절성 패턴 기반 조정
        seasonal_factor = self._calculate_seasonal_factor(market_data)
        
        # 4. 순환 패턴 기반 조정
        cycle_factor = self._calculate_cycle_factor(market_data)
        
        # 5. 통합 윈도우 계산
        combined_factor = (
            0.4 * volatility_factor + 
            0.3 * crisis_factor + 
            0.2 * seasonal_factor + 
            0.1 * cycle_factor
        )
        
        adaptive_window = int(self.base_window * combined_factor)
        adaptive_window = np.clip(adaptive_window, self.min_window, self.max_window)
        
        # 히스토리 업데이트
        self.window_history.append(adaptive_window)
        self.volatility_history.append(current_volatility)
        
        return adaptive_window
    
    def _calculate_volatility_factor(self, current_volatility: float) -> float:
        """
        변동성 기반 인수 계산
        """
        if current_volatility > self.volatility_threshold * 2:
            return 0.5  # 높은 변동성 시 짧은 윈도우
        elif current_volatility > self.volatility_threshold:
            return 0.8  # 중간 변동성
        else:
            return 1.2  # 낮은 변동성 시 긴 윈도우
    
    def _calculate_crisis_factor(self, crisis_level: float) -> float:
        """
        위기 수준 기반 인수 계산
        """
        if crisis_level > 0.7:
            return 0.6  # 위기 시 짧은 윈도우
        elif crisis_level > 0.4:
            return 0.8
        else:
            return 1.0
    
    def _calculate_seasonal_factor(self, market_data: pd.DataFrame) -> float:
        """
        계절성 인수 계산
        """
        if len(market_data) < 252:
            return 1.0
        
        # 현재 월 추출
        current_month = market_data.index[-1].month
        
        # 월별 변동성 패턴 분석
        monthly_volatilities = {}
        for month in range(1, 13):
            month_data = market_data[market_data.index.month == month]
            if len(month_data) > 10:
                monthly_vol = month_data.pct_change().std().mean()
                monthly_volatilities[month] = monthly_vol
        
        if current_month in monthly_volatilities and len(monthly_volatilities) > 6:
            current_vol = monthly_volatilities[current_month]
            avg_vol = np.mean(list(monthly_volatilities.values()))
            
            # 현재 월의 변동성이 평균보다 높으면 짧은 윈도우
            if current_vol > avg_vol * 1.2:
                return 0.8
            elif current_vol < avg_vol * 0.8:
                return 1.2
        
        return 1.0
    
    def _calculate_cycle_factor(self, market_data: pd.DataFrame) -> float:
        """
        순환 패턴 기반 인수 계산
        """
        cycle_info = self.cycle_detector.detect_cycles(market_data)
        
        if cycle_info['in_cycle']:
            cycle_phase = cycle_info['phase']
            
            # 순환의 전환점 근처에서는 짧은 윈도우
            if cycle_phase in ['peak', 'trough']:
                return 0.7
            else:
                return 1.0
        
        return 1.0
    
    def get_multi_scale_windows(self, base_window: int) -> List[int]:
        """
        다중 스케일 윈도우 생성
        """
        return [
            base_window // 4,     # 단기
            base_window // 2,     # 중기
            base_window,          # 기본
            base_window * 2,      # 장기
            base_window * 4       # 초장기
        ]


class CyclicalPatternDetector:
    """
    순환 패턴 감지기
    - 시장 사이클 감지
    - 계절성 패턴 분석
    - 주기적 행동 예측
    """
    
    def __init__(self):
        self.cycle_history = deque(maxlen=1000)
        self.detected_cycles = {}
        
    def detect_cycles(self, market_data: pd.DataFrame) -> Dict:
        """
        순환 패턴 감지
        """
        if len(market_data) < 100:
            return {'in_cycle': False, 'phase': 'unknown'}
        
        # 가격 데이터를 수익률로 변환
        returns = market_data.pct_change().dropna()
        
        # 1. 단기 사이클 감지 (5-20일)
        short_cycle = self._detect_short_cycle(returns)
        
        # 2. 중기 사이클 감지 (20-60일)
        medium_cycle = self._detect_medium_cycle(returns)
        
        # 3. 장기 사이클 감지 (60-252일)
        long_cycle = self._detect_long_cycle(returns)
        
        # 4. 통합 사이클 정보
        cycle_info = {
            'in_cycle': any([short_cycle['detected'], medium_cycle['detected'], long_cycle['detected']]),
            'short_cycle': short_cycle,
            'medium_cycle': medium_cycle,
            'long_cycle': long_cycle,
            'phase': self._determine_overall_phase(short_cycle, medium_cycle, long_cycle)
        }
        
        self.cycle_history.append(cycle_info)
        
        return cycle_info
    
    def _detect_short_cycle(self, returns: pd.DataFrame) -> Dict:
        """
        단기 사이클 감지 (5-20일)
        """
        if len(returns) < 40:
            return {'detected': False, 'period': 0, 'phase': 'unknown'}
        
        # 최근 40일 데이터 사용
        recent_returns = returns.tail(40)
        cumulative_returns = recent_returns.cumsum()
        
        # 피크와 트러프 찾기
        peaks, troughs = self._find_peaks_troughs(cumulative_returns.mean(axis=1))
        
        if len(peaks) >= 2 and len(troughs) >= 2:
            # 평균 사이클 길이 계산
            peak_intervals = np.diff(peaks)
            trough_intervals = np.diff(troughs)
            
            avg_cycle_length = np.mean(list(peak_intervals) + list(trough_intervals))
            
            # 5-20일 범위의 사이클인지 확인
            if 5 <= avg_cycle_length <= 20:
                current_phase = self._determine_current_phase(
                    cumulative_returns.mean(axis=1), peaks, troughs
                )
                
                return {
                    'detected': True,
                    'period': int(avg_cycle_length),
                    'phase': current_phase,
                    'strength': self._calculate_cycle_strength(cumulative_returns.mean(axis=1))
                }
        
        return {'detected': False, 'period': 0, 'phase': 'unknown', 'strength': 0.0}
    
    def _detect_medium_cycle(self, returns: pd.DataFrame) -> Dict:
        """
        중기 사이클 감지 (20-60일)
        """
        if len(returns) < 120:
            return {'detected': False, 'period': 0, 'phase': 'unknown'}
        
        # 최근 120일 데이터 사용
        recent_returns = returns.tail(120)
        
        # 20일 이동평균 기반 분석
        ma_20 = recent_returns.rolling(20).mean()
        ma_signal = ma_20.mean(axis=1)
        
        # 피크와 트러프 찾기
        peaks, troughs = self._find_peaks_troughs(ma_signal)
        
        if len(peaks) >= 2 and len(troughs) >= 2:
            # 평균 사이클 길이 계산
            peak_intervals = np.diff(peaks)
            trough_intervals = np.diff(troughs)
            
            avg_cycle_length = np.mean(list(peak_intervals) + list(trough_intervals))
            
            # 20-60일 범위의 사이클인지 확인
            if 20 <= avg_cycle_length <= 60:
                current_phase = self._determine_current_phase(ma_signal, peaks, troughs)
                
                return {
                    'detected': True,
                    'period': int(avg_cycle_length),
                    'phase': current_phase,
                    'strength': self._calculate_cycle_strength(ma_signal)
                }
        
        return {'detected': False, 'period': 0, 'phase': 'unknown', 'strength': 0.0}
    
    def _detect_long_cycle(self, returns: pd.DataFrame) -> Dict:
        """
        장기 사이클 감지 (60-252일)
        """
        if len(returns) < 504:  # 최소 2년 데이터 필요
            return {'detected': False, 'period': 0, 'phase': 'unknown'}
        
        # 최근 504일 데이터 사용
        recent_returns = returns.tail(504)
        
        # 60일 이동평균 기반 분석
        ma_60 = recent_returns.rolling(60).mean()
        ma_signal = ma_60.mean(axis=1)
        
        # 피크와 트러프 찾기
        peaks, troughs = self._find_peaks_troughs(ma_signal)
        
        if len(peaks) >= 2 and len(troughs) >= 2:
            # 평균 사이클 길이 계산
            peak_intervals = np.diff(peaks)
            trough_intervals = np.diff(troughs)
            
            avg_cycle_length = np.mean(list(peak_intervals) + list(trough_intervals))
            
            # 60-252일 범위의 사이클인지 확인
            if 60 <= avg_cycle_length <= 252:
                current_phase = self._determine_current_phase(ma_signal, peaks, troughs)
                
                return {
                    'detected': True,
                    'period': int(avg_cycle_length),
                    'phase': current_phase,
                    'strength': self._calculate_cycle_strength(ma_signal)
                }
        
        return {'detected': False, 'period': 0, 'phase': 'unknown', 'strength': 0.0}
    
    def _find_peaks_troughs(self, signal: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        피크와 트러프 찾기
        """
        signal_values = signal.values
        
        # 단순한 피크/트러프 감지
        peaks = []
        troughs = []
        
        for i in range(1, len(signal_values) - 1):
            if signal_values[i] > signal_values[i-1] and signal_values[i] > signal_values[i+1]:
                peaks.append(i)
            elif signal_values[i] < signal_values[i-1] and signal_values[i] < signal_values[i+1]:
                troughs.append(i)
        
        return np.array(peaks), np.array(troughs)
    
    def _determine_current_phase(self, signal: pd.Series, peaks: np.ndarray, troughs: np.ndarray) -> str:
        """
        현재 사이클 단계 결정
        """
        current_idx = len(signal) - 1
        
        # 가장 최근 피크와 트러프 찾기
        recent_peaks = peaks[peaks <= current_idx]
        recent_troughs = troughs[troughs <= current_idx]
        
        if len(recent_peaks) == 0 and len(recent_troughs) == 0:
            return 'unknown'
        
        # 마지막 피크와 트러프 비교
        last_peak = recent_peaks[-1] if len(recent_peaks) > 0 else -1
        last_trough = recent_troughs[-1] if len(recent_troughs) > 0 else -1
        
        if last_peak > last_trough:
            # 피크 이후 -> 하락 단계
            return 'decline'
        elif last_trough > last_peak:
            # 트러프 이후 -> 상승 단계
            return 'rise'
        else:
            return 'unknown'
    
    def _calculate_cycle_strength(self, signal: pd.Series) -> float:
        """
        사이클 강도 계산
        """
        if len(signal) < 10:
            return 0.0
        
        # 신호의 정규화된 표준편차
        normalized_std = signal.std() / (abs(signal.mean()) + 1e-8)
        
        # 자기상관을 통한 주기성 강도
        autocorr = signal.autocorr(lag=1)
        
        # 강도 = 변동성 + 자기상관
        strength = (normalized_std + abs(autocorr)) / 2
        
        return np.clip(strength, 0.0, 1.0)
    
    def _determine_overall_phase(self, short_cycle: Dict, medium_cycle: Dict, long_cycle: Dict) -> str:
        """
        전체 사이클 단계 결정
        """
        phases = []
        strengths = []
        
        for cycle in [short_cycle, medium_cycle, long_cycle]:
            if cycle['detected']:
                phases.append(cycle['phase'])
                strengths.append(cycle['strength'])
        
        if not phases:
            return 'unknown'
        
        # 가중 평균을 통한 단계 결정
        phase_weights = {'peak': 1, 'rise': 0.5, 'decline': -0.5, 'trough': -1, 'unknown': 0}
        
        weighted_score = 0
        total_weight = 0
        
        for phase, strength in zip(phases, strengths):
            if phase in phase_weights:
                weighted_score += phase_weights[phase] * strength
                total_weight += strength
        
        if total_weight == 0:
            return 'unknown'
        
        avg_score = weighted_score / total_weight
        
        if avg_score > 0.5:
            return 'peak'
        elif avg_score > 0:
            return 'rise'
        elif avg_score > -0.5:
            return 'decline'
        else:
            return 'trough'


class TemporalStateEncoder:
    """
    시계열 특화 상태 인코더
    - 다중 시간 스케일 특성 추출
    - 시간적 어텐션 메커니즘
    - 순환 패턴 임베딩
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_scales: int = 5):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        # 다중 스케일 LSTM
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_dim, hidden_dim, batch_first=True)
            for _ in range(num_scales)
        ])
        
        # 시간적 어텐션
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # 순환 패턴 임베딩
        self.cycle_embedding = nn.Embedding(8, hidden_dim // 4)  # 8개 사이클 단계
        
        # 특성 융합
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 상태 예측기
        self.state_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
    def encode_temporal_state(self, 
                            multi_scale_data: List[torch.Tensor],
                            cycle_phase: str = 'unknown') -> Dict:
        """
        시계열 특화 상태 인코딩
        """
        # 1. 다중 스케일 LSTM 처리
        scale_features = []
        for i, (data, lstm) in enumerate(zip(multi_scale_data, self.lstm_layers)):
            if len(data.shape) == 2:
                data = data.unsqueeze(0)  # 배치 차원 추가
            
            lstm_out, (hidden, cell) = lstm(data)
            # 마지막 히든 스테이트 사용
            scale_features.append(hidden[-1].squeeze(0))
        
        # 2. 시간적 어텐션 적용
        stacked_features = torch.stack(scale_features, dim=1)
        attended_features = self.temporal_attention(stacked_features)
        
        # 3. 순환 패턴 임베딩
        cycle_phases = ['unknown', 'peak', 'rise', 'decline', 'trough', 'expansion', 'contraction', 'transition']
        cycle_idx = cycle_phases.index(cycle_phase) if cycle_phase in cycle_phases else 0
        cycle_emb = self.cycle_embedding(torch.tensor([cycle_idx]))
        
        # 4. 특성 융합
        concatenated = torch.cat([attended_features.flatten(), cycle_emb.squeeze(0)], dim=0)
        fused_features = self.feature_fusion(concatenated)
        
        # 5. 다음 상태 예측
        predicted_state = self.state_predictor(fused_features)
        
        return {
            'encoded_state': fused_features,
            'predicted_next_state': predicted_state,
            'scale_features': scale_features,
            'attention_weights': self.temporal_attention.last_attention_weights,
            'cycle_embedding': cycle_emb
        }
    
    def calculate_temporal_consistency(self, 
                                     past_states: List[torch.Tensor],
                                     current_state: torch.Tensor) -> float:
        """
        시간적 일관성 계산
        """
        if len(past_states) < 2:
            return 0.5
        
        # 과거 상태들 간의 유사도 계산
        similarities = []
        for i in range(len(past_states) - 1):
            similarity = F.cosine_similarity(
                past_states[i].unsqueeze(0),
                past_states[i+1].unsqueeze(0)
            ).item()
            similarities.append(similarity)
        
        # 현재 상태와 가장 최근 상태 간의 유사도
        current_similarity = F.cosine_similarity(
            past_states[-1].unsqueeze(0),
            current_state.unsqueeze(0)
        ).item()
        
        # 평균 유사도
        avg_similarity = (np.mean(similarities) + current_similarity) / 2
        
        return (avg_similarity + 1) / 2  # [-1, 1] -> [0, 1]


class TemporalAttention(nn.Module):
    """
    시간적 어텐션 메커니즘
    """
    
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.last_attention_weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, hidden_dim)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # 배치 차원 추가
        
        # 셀프 어텐션
        attn_output, attn_weights = self.attention(x, x, x)
        self.last_attention_weights = attn_weights
        
        # 가중 평균
        attended_output = attn_output.mean(dim=1)
        
        return attended_output


class ImmuneMemoryDynamics:
    """
    면역 메모리 동역학
    - 메모리 강화/약화 메커니즘
    - 시간에 따른 메모리 감쇠
    - 메모리 재활성화 패턴
    """
    
    def __init__(self, decay_rate: float = 0.01, reinforcement_strength: float = 0.1):
        self.decay_rate = decay_rate
        self.reinforcement_strength = reinforcement_strength
        
        # 메모리 상태 추적
        self.memory_states = deque(maxlen=1000)
        self.memory_activations = deque(maxlen=1000)
        
        # 메모리 계층 구조
        self.short_term_memory = deque(maxlen=100)  # 1-7일
        self.medium_term_memory = deque(maxlen=500)  # 1주-3개월
        self.long_term_memory = deque(maxlen=2000)  # 3개월 이상
        
    def update_memory_dynamics(self, 
                             current_experience: Dict,
                             reward: float,
                             time_delta: int = 1) -> Dict:
        """
        메모리 동역학 업데이트
        """
        # 1. 메모리 감쇠
        self._apply_memory_decay(time_delta)
        
        # 2. 새로운 경험 저장
        memory_strength = self._calculate_initial_memory_strength(reward)
        
        memory_entry = {
            'experience': current_experience,
            'strength': memory_strength,
            'timestamp': len(self.memory_states),
            'activation_count': 0,
            'last_activation': len(self.memory_states)
        }
        
        # 3. 메모리 계층별 저장
        self._store_in_memory_layers(memory_entry)
        
        # 4. 메모리 재활성화 확인
        reactivated_memories = self._check_memory_reactivation(current_experience)
        
        # 5. 메모리 강화
        self._reinforce_memories(reactivated_memories)
        
        return {
            'new_memory': memory_entry,
            'reactivated_memories': reactivated_memories,
            'memory_statistics': self._calculate_memory_statistics()
        }
    
    def _apply_memory_decay(self, time_delta: int):
        """
        메모리 감쇠 적용
        """
        decay_factor = np.exp(-self.decay_rate * time_delta)
        
        # 각 메모리 계층별 감쇠
        for memory_layer in [self.short_term_memory, self.medium_term_memory, self.long_term_memory]:
            for memory in memory_layer:
                memory['strength'] *= decay_factor
                
                # 너무 약한 메모리는 제거
                if memory['strength'] < 0.01:
                    memory_layer.remove(memory)
    
    def _calculate_initial_memory_strength(self, reward: float) -> float:
        """
        초기 메모리 강도 계산
        """
        # 보상의 크기에 비례하여 메모리 강도 결정
        base_strength = 0.5
        reward_bonus = np.tanh(abs(reward) * 5) * 0.4
        
        # 부정적 경험에 더 강한 메모리
        if reward < 0:
            reward_bonus *= 1.5
        
        return np.clip(base_strength + reward_bonus, 0.1, 1.0)
    
    def _store_in_memory_layers(self, memory_entry: Dict):
        """
        메모리 계층별 저장
        """
        # 모든 경험은 단기 메모리에 저장
        self.short_term_memory.append(memory_entry.copy())
        
        # 강한 메모리는 중기 메모리에 저장
        if memory_entry['strength'] > 0.6:
            self.medium_term_memory.append(memory_entry.copy())
        
        # 매우 강한 메모리는 장기 메모리에 저장
        if memory_entry['strength'] > 0.8:
            self.long_term_memory.append(memory_entry.copy())
        
        # 전체 메모리 상태 업데이트
        self.memory_states.append(memory_entry)
    
    def _check_memory_reactivation(self, current_experience: Dict) -> List[Dict]:
        """
        메모리 재활성화 확인
        """
        reactivated = []
        
        # 각 메모리 계층에서 유사한 경험 찾기
        for memory_layer in [self.long_term_memory, self.medium_term_memory, self.short_term_memory]:
            for memory in memory_layer:
                similarity = self._calculate_experience_similarity(
                    current_experience, memory['experience']
                )
                
                # 높은 유사도일 때 재활성화
                if similarity > 0.7:
                    memory['activation_count'] += 1
                    memory['last_activation'] = len(self.memory_states)
                    reactivated.append(memory)
        
        return reactivated
    
    def _calculate_experience_similarity(self, exp1: Dict, exp2: Dict) -> float:
        """
        경험 간 유사도 계산
        """
        # 간단한 유사도 계산 (실제로는 더 정교한 방법 필요)
        if 'state' in exp1 and 'state' in exp2:
            state1 = np.array(exp1['state']) if isinstance(exp1['state'], list) else exp1['state']
            state2 = np.array(exp2['state']) if isinstance(exp2['state'], list) else exp2['state']
            
            # 코사인 유사도
            dot_product = np.dot(state1, state2)
            norm1 = np.linalg.norm(state1)
            norm2 = np.linalg.norm(state2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        return 0.0
    
    def _reinforce_memories(self, reactivated_memories: List[Dict]):
        """
        메모리 강화
        """
        for memory in reactivated_memories:
            # 재활성화 횟수에 따른 강화
            reinforcement = self.reinforcement_strength * np.log(memory['activation_count'] + 1)
            memory['strength'] = min(memory['strength'] + reinforcement, 1.0)
    
    def _calculate_memory_statistics(self) -> Dict:
        """
        메모리 통계 계산
        """
        all_memories = (list(self.short_term_memory) + 
                       list(self.medium_term_memory) + 
                       list(self.long_term_memory))
        
        if not all_memories:
            return {
                'total_memories': 0,
                'avg_strength': 0.0,
                'memory_distribution': {'short': 0, 'medium': 0, 'long': 0}
            }
        
        strengths = [m['strength'] for m in all_memories]
        
        return {
            'total_memories': len(all_memories),
            'avg_strength': np.mean(strengths),
            'memory_distribution': {
                'short': len(self.short_term_memory),
                'medium': len(self.medium_term_memory),
                'long': len(self.long_term_memory)
            },
            'strong_memories': sum(1 for s in strengths if s > 0.7),
            'weak_memories': sum(1 for s in strengths if s < 0.3)
        }
    
    def retrieve_relevant_memories(self, query_experience: Dict, top_k: int = 5) -> List[Dict]:
        """
        관련 메모리 검색
        """
        all_memories = (list(self.long_term_memory) + 
                       list(self.medium_term_memory) + 
                       list(self.short_term_memory))
        
        if not all_memories:
            return []
        
        # 유사도 계산
        similarities = []
        for memory in all_memories:
            similarity = self._calculate_experience_similarity(
                query_experience, memory['experience']
            )
            # 메모리 강도와 유사도를 결합
            relevance_score = similarity * memory['strength']
            similarities.append((relevance_score, memory))
        
        # 상위 k개 선택
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in similarities[:top_k]]


class BIPDTemporalEngine:
    """
    BIPD 시계열 엔진
    모든 시계열 특화 컴포넌트를 통합
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 컴포넌트 초기화
        self.window_manager = AdaptiveTimeWindowManager()
        self.cycle_detector = CyclicalPatternDetector()
        self.state_encoder = TemporalStateEncoder(input_dim, hidden_dim)
        self.memory_dynamics = ImmuneMemoryDynamics()
        
        # 시계열 상태 히스토리
        self.state_history = deque(maxlen=1000)
        
    def process_temporal_data(self, 
                            market_data: pd.DataFrame,
                            current_features: np.ndarray,
                            current_volatility: float,
                            crisis_level: float,
                            reward: float = 0.0) -> Dict:
        """
        시계열 데이터 종합 처리
        """
        # 1. 적응적 시간 윈도우 계산
        adaptive_window = self.window_manager.calculate_adaptive_window(
            market_data, current_volatility, crisis_level
        )
        
        # 2. 다중 스케일 윈도우 생성
        multi_scale_windows = self.window_manager.get_multi_scale_windows(adaptive_window)
        
        # 3. 순환 패턴 감지
        cycle_info = self.cycle_detector.detect_cycles(market_data)
        
        # 4. 다중 스케일 데이터 준비
        multi_scale_data = self._prepare_multi_scale_data(
            market_data, multi_scale_windows
        )
        
        # 5. 시계열 상태 인코딩
        encoded_state = self.state_encoder.encode_temporal_state(
            multi_scale_data, cycle_info['phase']
        )
        
        # 6. 메모리 동역학 업데이트
        current_experience = {
            'state': current_features,
            'encoded_state': encoded_state['encoded_state'].detach().numpy(),
            'cycle_phase': cycle_info['phase'],
            'crisis_level': crisis_level
        }
        
        memory_update = self.memory_dynamics.update_memory_dynamics(
            current_experience, reward
        )
        
        # 7. 시간적 일관성 계산
        past_encoded_states = [s['encoded_state'] for s in self.state_history[-10:]]
        temporal_consistency = self.state_encoder.calculate_temporal_consistency(
            past_encoded_states, encoded_state['encoded_state']
        ) if past_encoded_states else 0.5
        
        # 8. 상태 히스토리 업데이트
        self.state_history.append({
            'encoded_state': encoded_state['encoded_state'].detach(),
            'cycle_info': cycle_info,
            'adaptive_window': adaptive_window,
            'timestamp': len(self.state_history)
        })
        
        return {
            'adaptive_window': adaptive_window,
            'multi_scale_windows': multi_scale_windows,
            'cycle_info': cycle_info,
            'encoded_state': encoded_state,
            'memory_update': memory_update,
            'temporal_consistency': temporal_consistency,
            'temporal_insights': self._generate_temporal_insights()
        }
    
    def _prepare_multi_scale_data(self, 
                                market_data: pd.DataFrame, 
                                windows: List[int]) -> List[torch.Tensor]:
        """
        다중 스케일 데이터 준비
        """
        multi_scale_data = []
        
        for window in windows:
            if len(market_data) >= window:
                # 해당 윈도우 크기의 데이터 추출
                window_data = market_data.tail(window)
                
                # 수익률 계산
                returns = window_data.pct_change().dropna()
                
                # 텐서 변환
                if not returns.empty:
                    tensor_data = torch.tensor(returns.values, dtype=torch.float32)
                    multi_scale_data.append(tensor_data)
                else:
                    # 빈 데이터 처리
                    multi_scale_data.append(torch.zeros(1, self.input_dim))
            else:
                # 데이터가 부족한 경우
                multi_scale_data.append(torch.zeros(1, self.input_dim))
        
        return multi_scale_data
    
    def _generate_temporal_insights(self) -> Dict:
        """
        시계열 인사이트 생성
        """
        if len(self.state_history) < 10:
            return {'status': 'insufficient_data'}
        
        recent_states = list(self.state_history)[-10:]
        
        # 윈도우 적응 패턴
        windows = [s['adaptive_window'] for s in recent_states]
        window_trend = np.polyfit(range(len(windows)), windows, 1)[0]
        
        # 사이클 안정성
        cycle_phases = [s['cycle_info']['phase'] for s in recent_states]
        phase_changes = sum(1 for i in range(1, len(cycle_phases)) 
                           if cycle_phases[i] != cycle_phases[i-1])
        cycle_stability = 1.0 - (phase_changes / len(cycle_phases))
        
        # 메모리 활용도
        memory_stats = self.memory_dynamics._calculate_memory_statistics()
        
        return {
            'window_adaptation': {
                'current_window': windows[-1],
                'trend': window_trend,
                'volatility': np.std(windows)
            },
            'cycle_stability': cycle_stability,
            'current_phase': cycle_phases[-1],
            'memory_utilization': memory_stats,
            'temporal_health': (cycle_stability + min(memory_stats['avg_strength'], 1.0)) / 2
        }
    
    def get_temporal_recommendations(self) -> List[str]:
        """
        시계열 기반 권고사항
        """
        insights = self._generate_temporal_insights()
        recommendations = []
        
        if insights['status'] == 'insufficient_data':
            return ['Need more data for temporal analysis']
        
        # 윈도우 적응 권고
        if insights['window_adaptation']['trend'] > 5:
            recommendations.append('Increasing window size trend detected. Market may be becoming more stable.')
        elif insights['window_adaptation']['trend'] < -5:
            recommendations.append('Decreasing window size trend detected. Market volatility may be increasing.')
        
        # 사이클 권고
        if insights['cycle_stability'] < 0.5:
            recommendations.append('High cycle instability detected. Consider more conservative strategies.')
        
        current_phase = insights['current_phase']
        if current_phase == 'peak':
            recommendations.append('Market cycle at peak. Consider taking profits or reducing exposure.')
        elif current_phase == 'trough':
            recommendations.append('Market cycle at trough. Consider increasing exposure or buying opportunities.')
        
        # 메모리 권고
        memory_strength = insights['memory_utilization']['avg_strength']
        if memory_strength < 0.3:
            recommendations.append('Low memory strength. System may not be learning effectively from past experiences.')
        elif memory_strength > 0.8:
            recommendations.append('High memory strength. System is effectively learning from past experiences.')
        
        # 전체 시계열 건강성 권고
        temporal_health = insights['temporal_health']
        if temporal_health < 0.4:
            recommendations.append('Poor temporal health. Consider system parameter adjustments.')
        elif temporal_health > 0.8:
            recommendations.append('Excellent temporal health. System is well-adapted to current market conditions.')
        
        return recommendations if recommendations else ['No specific temporal recommendations at this time.']