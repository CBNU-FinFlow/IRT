"""
BIPD 보상 시스템
생체면역 시스템에서 영감받은 정교한 보상 함수 설계

논문 기여도:
1. 다층 보상 구조 (즉시 보상 + 장기 보상 + 면역 보상)
2. 적응적 리스크 조정 메커니즘
3. 시계열 특화 성과 평가
4. 면역 시스템 건강성 평가
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque
import torch


class AdaptiveRiskMetrics:
    """
    적응적 리스크 측정 지표
    - 동적 VaR (Value at Risk)
    - 조건부 VaR (Expected Shortfall)
    - 최대 낙폭 추적
    - 변동성 체제 변화 감지
    """
    
    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        self.return_history = deque(maxlen=lookback_window)
        self.volatility_history = deque(maxlen=lookback_window)
        self.drawdown_history = deque(maxlen=lookback_window)
        
        # 변동성 체제 추적
        self.volatility_regimes = ['low', 'medium', 'high']
        self.current_regime = 'medium'
        self.regime_history = deque(maxlen=100)
        
    def update_metrics(self, returns: np.ndarray, portfolio_value: float) -> Dict:
        """
        리스크 지표 업데이트
        """
        current_return = returns[-1] if len(returns) > 0 else 0.0
        self.return_history.append(current_return)
        
        if len(self.return_history) < 30:
            return self._default_metrics()
        
        returns_array = np.array(self.return_history)
        
        # 변동성 계산
        volatility = np.std(returns_array) * np.sqrt(252)
        self.volatility_history.append(volatility)
        
        # VaR 계산 (95% 신뢰구간)
        var_95 = np.percentile(returns_array, 5)
        
        # 조건부 VaR (Expected Shortfall)
        es_95 = np.mean(returns_array[returns_array <= var_95])
        
        # 최대 낙폭 계산
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        self.drawdown_history.append(max_drawdown)
        
        # 변동성 체제 감지
        self._detect_volatility_regime(volatility)
        
        # 샤프 비율
        sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
        
        # 칼마 비율
        calmar_ratio = np.mean(returns_array) * 252 / (abs(max_drawdown) + 1e-8)
        
        return {
            'var_95': var_95,
            'expected_shortfall': es_95,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'volatility_regime': self.current_regime,
            'regime_stability': self._calculate_regime_stability()
        }
    
    def _detect_volatility_regime(self, current_volatility: float):
        """
        변동성 체제 감지
        """
        if len(self.volatility_history) < 20:
            return
            
        vol_history = np.array(self.volatility_history)
        vol_mean = np.mean(vol_history)
        vol_std = np.std(vol_history)
        
        # 변동성 체제 분류
        if current_volatility < vol_mean - 0.5 * vol_std:
            new_regime = 'low'
        elif current_volatility > vol_mean + 0.5 * vol_std:
            new_regime = 'high'
        else:
            new_regime = 'medium'
        
        if new_regime != self.current_regime:
            self.current_regime = new_regime
            
        self.regime_history.append(self.current_regime)
    
    def _calculate_regime_stability(self) -> float:
        """
        체제 안정성 계산
        """
        if len(self.regime_history) < 10:
            return 0.5
            
        # 최근 10개 체제에서 변화 횟수
        changes = sum(1 for i in range(1, len(list(self.regime_history)[-10:])) 
                     if list(self.regime_history)[-10:][i] != list(self.regime_history)[-10:][i-1])
        
        # 안정성 = 1 - (변화 횟수 / 최대 가능 변화 횟수)
        stability = 1.0 - (changes / 9.0)
        return stability
    
    def _default_metrics(self) -> Dict:
        """
        기본 지표 반환
        """
        return {
            'var_95': 0.0,
            'expected_shortfall': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'calmar_ratio': 0.0,
            'volatility_regime': 'medium',
            'regime_stability': 0.5
        }


class ImmuneSystemHealthMetrics:
    """
    면역 시스템 건강성 평가
    - T-Cell 다양성 및 적응성
    - B-Cell 전문화 효율성
    - 메모리 시스템 활용도
    - 세포간 협력 수준
    """
    
    def __init__(self):
        self.tcell_diversity_history = deque(maxlen=100)
        self.bcell_specialization_history = deque(maxlen=100)
        self.memory_utilization_history = deque(maxlen=100)
        self.cooperation_history = deque(maxlen=100)
        
    def evaluate_system_health(self, agent_diagnostics: Dict) -> Dict:
        """
        면역 시스템 건강성 평가
        """
        # T-Cell 다양성 평가
        tcell_diversity = self._calculate_tcell_diversity(agent_diagnostics['tcell_diagnostics'])
        self.tcell_diversity_history.append(tcell_diversity)
        
        # B-Cell 전문화 효율성 평가
        bcell_efficiency = self._calculate_bcell_efficiency(agent_diagnostics['bcell_diagnostics'])
        self.bcell_specialization_history.append(bcell_efficiency)
        
        # 메모리 활용도 평가
        memory_utilization = self._calculate_memory_utilization(agent_diagnostics['memory_diagnostics'])
        self.memory_utilization_history.append(memory_utilization)
        
        # 협력 수준 평가
        cooperation_level = self._calculate_cooperation_level(agent_diagnostics)
        self.cooperation_history.append(cooperation_level)
        
        # 전체 건강성 점수
        health_score = (
            0.3 * tcell_diversity +
            0.3 * bcell_efficiency +
            0.2 * memory_utilization +
            0.2 * cooperation_level
        )
        
        return {
            'tcell_diversity': tcell_diversity,
            'bcell_efficiency': bcell_efficiency,
            'memory_utilization': memory_utilization,
            'cooperation_level': cooperation_level,
            'overall_health': health_score,
            'health_trend': self._calculate_health_trend()
        }
    
    def _calculate_tcell_diversity(self, tcell_diagnostics: List[Dict]) -> float:
        """
        T-Cell 다양성 계산
        """
        if not tcell_diagnostics:
            return 0.0
            
        # 민감도 다양성
        sensitivities = [diag['sensitivity'] for diag in tcell_diagnostics]
        sensitivity_diversity = np.std(sensitivities) / (np.mean(sensitivities) + 1e-8)
        
        # 성과 다양성
        performances = [diag['avg_performance'] for diag in tcell_diagnostics]
        performance_diversity = 1.0 - (np.std(performances) / (np.mean(np.abs(performances)) + 1e-8))
        
        # 메모리 크기 다양성
        memory_sizes = [diag['memory_size'] for diag in tcell_diagnostics]
        memory_diversity = np.std(memory_sizes) / (np.mean(memory_sizes) + 1e-8)
        
        diversity_score = (sensitivity_diversity + performance_diversity + memory_diversity) / 3.0
        return np.clip(diversity_score, 0.0, 1.0)
    
    def _calculate_bcell_efficiency(self, bcell_diagnostics: List[Dict]) -> float:
        """
        B-Cell 전문화 효율성 계산
        """
        if not bcell_diagnostics:
            return 0.0
            
        # 전문화별 성과 차이
        specialization_performance = {}
        for diag in bcell_diagnostics:
            spec = diag['specialization']
            perf = diag['avg_specialization_performance']
            if spec not in specialization_performance:
                specialization_performance[spec] = []
            specialization_performance[spec].append(perf)
        
        # 전문화 효율성 = 전문화 간 성과 차이가 클수록 좋음
        if len(specialization_performance) < 2:
            return 0.5
            
        spec_averages = [np.mean(perfs) for perfs in specialization_performance.values()]
        efficiency = np.std(spec_averages) / (np.mean(np.abs(spec_averages)) + 1e-8)
        
        # 경험 축적 정도
        experience_counts = [diag['experience_count'] for diag in bcell_diagnostics]
        experience_factor = np.mean(experience_counts) / 1000.0  # 정규화
        
        total_efficiency = 0.7 * efficiency + 0.3 * experience_factor
        return np.clip(total_efficiency, 0.0, 1.0)
    
    def _calculate_memory_utilization(self, memory_diagnostics: Dict) -> float:
        """
        메모리 활용도 계산
        """
        episodic_usage = memory_diagnostics['episodic_memory_size'] / 50000.0  # 최대 크기로 정규화
        semantic_richness = min(memory_diagnostics['semantic_patterns'] / 20.0, 1.0)  # 20개 패턴이 최대
        working_memory_activity = memory_diagnostics['working_memory_size'] / 100.0
        
        utilization = (episodic_usage + semantic_richness + working_memory_activity) / 3.0
        return np.clip(utilization, 0.0, 1.0)
    
    def _calculate_cooperation_level(self, agent_diagnostics: Dict) -> float:
        """
        세포간 협력 수준 계산
        """
        system_state = agent_diagnostics['system_state']
        
        # 시스템 상태의 안정성
        crisis_stability = 1.0 - abs(system_state['crisis_level'] - 0.5)  # 0.5 근처가 안정
        
        # 활성 B-Cell 수 (협력의 지표)
        active_bcells_ratio = len(system_state['active_bcells']) / 5.0  # 5개 중 몇 개가 활성
        
        # 전체 성과 (협력의 결과)
        overall_performance = agent_diagnostics['overall_performance']
        performance_factor = np.tanh(overall_performance * 10)  # sigmoid 형태로 변환
        
        cooperation = (crisis_stability + active_bcells_ratio + performance_factor) / 3.0
        return np.clip(cooperation, 0.0, 1.0)
    
    def _calculate_health_trend(self) -> float:
        """
        건강성 추세 계산
        """
        if len(self.tcell_diversity_history) < 10:
            return 0.0
            
        # 최근 10개 값의 추세
        recent_values = list(self.tcell_diversity_history)[-10:]
        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        
        return np.clip(trend, -1.0, 1.0)


class BIPDRewardSystem:
    """
    BIPD 보상 시스템
    다층 보상 구조로 구성된 정교한 보상 함수
    """
    
    def __init__(self, lookback_window: int = 252):
        self.risk_metrics = AdaptiveRiskMetrics(lookback_window)
        self.health_metrics = ImmuneSystemHealthMetrics()
        
        # 보상 가중치 (동적으로 조정)
        self.reward_weights = {
            'immediate_return': 0.4,
            'risk_adjusted_return': 0.3,
            'immune_health': 0.2,
            'long_term_stability': 0.1
        }
        
        # 보상 히스토리
        self.reward_history = deque(maxlen=1000)
        self.component_history = {
            'immediate': deque(maxlen=1000),
            'risk_adjusted': deque(maxlen=1000),
            'immune_health': deque(maxlen=1000),
            'long_term': deque(maxlen=1000)
        }
        
    def calculate_reward(self, 
                        portfolio_return: float,
                        portfolio_returns: np.ndarray,
                        portfolio_value: float,
                        agent_diagnostics: Dict,
                        market_context: Dict) -> Dict:
        """
        종합 보상 계산
        """
        # 1. 즉시 보상 (수익률 기반)
        immediate_reward = self._calculate_immediate_reward(portfolio_return, market_context)
        
        # 2. 리스크 조정 보상
        risk_adjusted_reward = self._calculate_risk_adjusted_reward(
            portfolio_return, portfolio_returns, portfolio_value, market_context
        )
        
        # 3. 면역 시스템 건강성 보상
        immune_health_reward = self._calculate_immune_health_reward(agent_diagnostics)
        
        # 4. 장기 안정성 보상
        long_term_reward = self._calculate_long_term_reward(portfolio_returns)
        
        # 5. 동적 가중치 조정
        self._adjust_reward_weights(market_context)
        
        # 6. 최종 보상 계산
        total_reward = (
            self.reward_weights['immediate_return'] * immediate_reward +
            self.reward_weights['risk_adjusted_return'] * risk_adjusted_reward +
            self.reward_weights['immune_health'] * immune_health_reward +
            self.reward_weights['long_term_stability'] * long_term_reward
        )
        
        # 보상 히스토리 업데이트
        self.reward_history.append(total_reward)
        self.component_history['immediate'].append(immediate_reward)
        self.component_history['risk_adjusted'].append(risk_adjusted_reward)
        self.component_history['immune_health'].append(immune_health_reward)
        self.component_history['long_term'].append(long_term_reward)
        
        return {
            'total_reward': total_reward,
            'immediate_reward': immediate_reward,
            'risk_adjusted_reward': risk_adjusted_reward,
            'immune_health_reward': immune_health_reward,
            'long_term_reward': long_term_reward,
            'reward_weights': self.reward_weights.copy(),
            'reward_components_stats': self._get_reward_stats()
        }
    
    def _calculate_immediate_reward(self, portfolio_return: float, market_context: Dict) -> float:
        """
        즉시 보상 계산 (수익률 기반)
        """
        # 기본 수익률 보상
        base_reward = portfolio_return * 10  # 스케일링
        
        # 시장 상황 보정
        market_return = market_context.get('market_return', 0.0)
        alpha = portfolio_return - market_return  # 초과 수익률
        
        # 초과 수익률에 보너스
        alpha_bonus = alpha * 5 if alpha > 0 else alpha * 3  # 손실에 더 큰 페널티
        
        # 변동성 조정
        volatility = market_context.get('volatility', 0.02)
        vol_penalty = -volatility * 2  # 높은 변동성에 페널티
        
        immediate_reward = base_reward + alpha_bonus + vol_penalty
        
        return np.clip(immediate_reward, -1.0, 1.0)
    
    def _calculate_risk_adjusted_reward(self, 
                                      portfolio_return: float,
                                      portfolio_returns: np.ndarray,
                                      portfolio_value: float,
                                      market_context: Dict) -> float:
        """
        리스크 조정 보상 계산
        """
        # 리스크 지표 업데이트
        risk_metrics = self.risk_metrics.update_metrics(portfolio_returns, portfolio_value)
        
        # 샤프 비율 기반 보상
        sharpe_reward = np.tanh(risk_metrics['sharpe_ratio']) * 0.5
        
        # 최대 낙폭 페널티
        drawdown_penalty = risk_metrics['max_drawdown'] * 2
        
        # VaR 초과 페널티
        var_penalty = 0.0
        if portfolio_return < risk_metrics['var_95']:
            var_penalty = -0.3
        
        # 변동성 체제 적응 보상
        regime_bonus = 0.0
        if risk_metrics['regime_stability'] > 0.8:
            regime_bonus = 0.1
        
        # 칼마 비율 보상
        calmar_reward = np.tanh(risk_metrics['calmar_ratio']) * 0.3
        
        risk_adjusted_reward = (
            sharpe_reward + drawdown_penalty + var_penalty + 
            regime_bonus + calmar_reward
        )
        
        return np.clip(risk_adjusted_reward, -1.0, 1.0)
    
    def _calculate_immune_health_reward(self, agent_diagnostics: Dict) -> float:
        """
        면역 시스템 건강성 보상
        """
        health_metrics = self.health_metrics.evaluate_system_health(agent_diagnostics)
        
        # 전체 건강성 점수
        health_score = health_metrics['overall_health']
        
        # 건강성 추세 보상
        trend_bonus = health_metrics['health_trend'] * 0.1
        
        # 개별 구성 요소 보상
        diversity_bonus = health_metrics['tcell_diversity'] * 0.1
        efficiency_bonus = health_metrics['bcell_efficiency'] * 0.1
        memory_bonus = health_metrics['memory_utilization'] * 0.05
        cooperation_bonus = health_metrics['cooperation_level'] * 0.05
        
        immune_reward = (
            health_score * 0.7 + trend_bonus + diversity_bonus + 
            efficiency_bonus + memory_bonus + cooperation_bonus
        )
        
        return np.clip(immune_reward, 0.0, 1.0)
    
    def _calculate_long_term_reward(self, portfolio_returns: np.ndarray) -> float:
        """
        장기 안정성 보상
        """
        if len(portfolio_returns) < 50:
            return 0.0
            
        # 장기 수익률 추세
        long_term_returns = portfolio_returns[-50:]
        trend = np.polyfit(range(len(long_term_returns)), 
                          np.cumsum(long_term_returns), 1)[0]
        trend_reward = np.tanh(trend * 100) * 0.5
        
        # 수익률 일관성 (낮은 변동성)
        consistency_reward = 1.0 / (1.0 + np.std(long_term_returns) * 10)
        
        # 복합 성장률
        compound_growth = np.prod(1 + long_term_returns) ** (252 / len(long_term_returns)) - 1
        growth_reward = np.tanh(compound_growth * 5) * 0.3
        
        long_term_reward = trend_reward + consistency_reward + growth_reward
        
        return np.clip(long_term_reward, 0.0, 1.0)
    
    def _adjust_reward_weights(self, market_context: Dict):
        """
        시장 상황에 따른 동적 가중치 조정
        """
        volatility = market_context.get('volatility', 0.02)
        crisis_level = market_context.get('crisis_level', 0.0)
        
        # 고변동성 시장에서는 리스크 조정 보상 가중치 증가
        if volatility > 0.03:
            self.reward_weights['risk_adjusted_return'] = 0.4
            self.reward_weights['immediate_return'] = 0.3
        else:
            self.reward_weights['risk_adjusted_return'] = 0.3
            self.reward_weights['immediate_return'] = 0.4
        
        # 위기 상황에서는 면역 시스템 건강성 가중치 증가
        if crisis_level > 0.7:
            self.reward_weights['immune_health'] = 0.3
            self.reward_weights['long_term_stability'] = 0.05
        else:
            self.reward_weights['immune_health'] = 0.2
            self.reward_weights['long_term_stability'] = 0.1
        
        # 가중치 정규화
        total_weight = sum(self.reward_weights.values())
        for key in self.reward_weights:
            self.reward_weights[key] /= total_weight
    
    def _get_reward_stats(self) -> Dict:
        """
        보상 구성 요소 통계
        """
        stats = {}
        for component, history in self.component_history.items():
            if len(history) > 0:
                stats[component] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'trend': np.polyfit(range(len(history)), list(history), 1)[0] if len(history) > 1 else 0.0
                }
            else:
                stats[component] = {'mean': 0.0, 'std': 0.0, 'trend': 0.0}
        
        return stats
    
    def get_reward_analysis(self) -> Dict:
        """
        보상 시스템 분석 결과
        """
        if len(self.reward_history) < 10:
            return {'status': 'insufficient_data'}
        
        reward_array = np.array(self.reward_history)
        
        return {
            'total_reward_stats': {
                'mean': np.mean(reward_array),
                'std': np.std(reward_array),
                'trend': np.polyfit(range(len(reward_array)), reward_array, 1)[0],
                'recent_performance': np.mean(reward_array[-10:])
            },
            'component_contributions': self._get_reward_stats(),
            'current_weights': self.reward_weights.copy(),
            'reward_distribution': {
                'positive_ratio': np.mean(reward_array > 0),
                'percentiles': {
                    '5th': np.percentile(reward_array, 5),
                    '25th': np.percentile(reward_array, 25),
                    '50th': np.percentile(reward_array, 50),
                    '75th': np.percentile(reward_array, 75),
                    '95th': np.percentile(reward_array, 95)
                }
            }
        }