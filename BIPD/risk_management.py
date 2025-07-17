"""
BIPD 리스크 관리 시스템
생체면역 시스템에서 영감받은 다층 리스크 관리 메커니즘

논문 기여도:
1. 면역 시스템 기반 적응적 리스크 관리
2. 다중 시간 스케일 리스크 모니터링
3. 동적 포지션 사이징 및 헤징 전략
4. 시계열 특화 위기 예측 시스템
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


class CrisisDetectionSystem:
    """
    위기 감지 시스템
    - 다중 시간 스케일 이상 감지
    - 체제 변화 감지
    - 테일 리스크 모니터링
    """
    
    def __init__(self, lookback_periods: List[int] = [5, 20, 60]):
        self.lookback_periods = lookback_periods
        self.detectors = {}
        self.scalers = {}
        self.crisis_history = deque(maxlen=1000)
        
        # 각 시간 스케일별 이상 감지기 초기화
        for period in lookback_periods:
            self.detectors[period] = IsolationForest(
                contamination=0.1, 
                random_state=42,
                n_estimators=100
            )
            self.scalers[period] = StandardScaler()
        
        # 위기 임계값 (적응적으로 조정)
        self.crisis_thresholds = {period: -0.5 for period in lookback_periods}
        
        # 체제 변화 감지
        self.regime_detector = RegimeChangeDetector()
        
        # 테일 리스크 모니터링
        self.tail_risk_monitor = TailRiskMonitor()
        
    def detect_crisis(self, market_data: pd.DataFrame, 
                     current_features: np.ndarray) -> Dict:
        """
        다층 위기 감지
        """
        crisis_signals = {}
        overall_crisis_level = 0.0
        
        # 1. 다중 시간 스케일 이상 감지
        for period in self.lookback_periods:
            if len(market_data) >= period:
                # 시간 스케일별 특성 추출
                period_features = self._extract_period_features(market_data, period)
                
                # 이상 감지기 훈련 (충분한 데이터가 있을 때)
                if len(period_features) >= 50:
                    scaled_features = self.scalers[period].fit_transform(period_features)
                    self.detectors[period].fit(scaled_features)
                    
                    # 현재 상태 이상 점수 계산
                    current_scaled = self.scalers[period].transform(
                        current_features.reshape(1, -1)
                    )
                    anomaly_score = self.detectors[period].score_samples(current_scaled)[0]
                    
                    # 위기 여부 판단
                    is_crisis = anomaly_score < self.crisis_thresholds[period]
                    
                    crisis_signals[f'period_{period}'] = {
                        'anomaly_score': anomaly_score,
                        'is_crisis': is_crisis,
                        'threshold': self.crisis_thresholds[period],
                        'severity': max(0, (self.crisis_thresholds[period] - anomaly_score) / 0.5)
                    }
                    
                    # 전체 위기 수준에 기여
                    if is_crisis:
                        overall_crisis_level += crisis_signals[f'period_{period}']['severity'] / len(self.lookback_periods)
        
        # 2. 체제 변화 감지
        regime_change = self.regime_detector.detect_regime_change(market_data)
        if regime_change['regime_changed']:
            overall_crisis_level += 0.3
        
        # 3. 테일 리스크 모니터링
        tail_risk = self.tail_risk_monitor.assess_tail_risk(market_data)
        overall_crisis_level += tail_risk['risk_level'] * 0.2
        
        # 4. 위기 히스토리 업데이트
        crisis_info = {
            'timestamp': len(self.crisis_history),
            'overall_crisis_level': overall_crisis_level,
            'crisis_signals': crisis_signals,
            'regime_change': regime_change,
            'tail_risk': tail_risk
        }
        self.crisis_history.append(crisis_info)
        
        # 5. 적응적 임계값 조정
        self._adjust_thresholds()
        
        return crisis_info
    
    def _extract_period_features(self, market_data: pd.DataFrame, period: int) -> np.ndarray:
        """
        시간 스케일별 특성 추출
        """
        if len(market_data) < period:
            return np.array([])
        
        # 수익률 계산
        returns = market_data.pct_change().dropna()
        
        features_list = []
        for i in range(period, len(returns)):
            period_returns = returns.iloc[i-period:i]
            
            # 통계적 특성
            mean_return = period_returns.mean().mean()
            volatility = period_returns.std().mean()
            skewness = period_returns.skew().mean()
            kurtosis = period_returns.kurtosis().mean()
            
            # 상관관계 특성
            corr_matrix = period_returns.corr()
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            
            # 최대 낙폭
            cumulative_returns = (1 + period_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min().min()
            
            # VaR 및 CVaR
            var_95 = period_returns.quantile(0.05).mean()
            cvar_95 = period_returns[period_returns <= period_returns.quantile(0.05)].mean().mean()
            
            features = [
                mean_return, volatility, skewness, kurtosis,
                avg_correlation, max_drawdown, var_95, cvar_95
            ]
            features_list.append(features)
        
        return np.array(features_list)
    
    def _adjust_thresholds(self):
        """
        적응적 임계값 조정
        """
        if len(self.crisis_history) < 50:
            return
        
        # 최근 성과 기반 조정
        recent_crises = list(self.crisis_history)[-50:]
        false_positive_rate = sum(1 for c in recent_crises 
                                 if c['overall_crisis_level'] > 0.5) / len(recent_crises)
        
        # 너무 많은 위기 신호시 임계값 조정
        if false_positive_rate > 0.3:
            for period in self.lookback_periods:
                self.crisis_thresholds[period] *= 0.95  # 더 엄격하게
        elif false_positive_rate < 0.1:
            for period in self.lookback_periods:
                self.crisis_thresholds[period] *= 1.05  # 더 민감하게


class RegimeChangeDetector:
    """
    체제 변화 감지
    - 변동성 체제 변화
    - 상관관계 구조 변화
    - 트렌드 전환점 감지
    """
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.previous_regime = None
        self.regime_history = deque(maxlen=100)
        
    def detect_regime_change(self, market_data: pd.DataFrame) -> Dict:
        """
        체제 변화 감지
        """
        if len(market_data) < self.window_size * 2:
            return {'regime_changed': False, 'regime_type': 'unknown'}
        
        # 최근 두 기간의 특성 비교
        recent_data = market_data.iloc[-self.window_size:]
        previous_data = market_data.iloc[-self.window_size*2:-self.window_size]
        
        # 변동성 체제 변화 감지
        volatility_change = self._detect_volatility_regime_change(recent_data, previous_data)
        
        # 상관관계 체제 변화 감지
        correlation_change = self._detect_correlation_regime_change(recent_data, previous_data)
        
        # 트렌드 체제 변화 감지
        trend_change = self._detect_trend_regime_change(recent_data, previous_data)
        
        # 전체 체제 변화 판단
        regime_changed = (volatility_change['changed'] or 
                         correlation_change['changed'] or 
                         trend_change['changed'])
        
        # 현재 체제 분류
        current_regime = self._classify_current_regime(recent_data)
        
        if regime_changed:
            self.previous_regime = current_regime
        
        regime_info = {
            'regime_changed': regime_changed,
            'regime_type': current_regime,
            'volatility_change': volatility_change,
            'correlation_change': correlation_change,
            'trend_change': trend_change,
            'change_magnitude': self._calculate_change_magnitude(
                volatility_change, correlation_change, trend_change
            )
        }
        
        self.regime_history.append(regime_info)
        
        return regime_info
    
    def _detect_volatility_regime_change(self, recent_data: pd.DataFrame, 
                                       previous_data: pd.DataFrame) -> Dict:
        """
        변동성 체제 변화 감지
        """
        recent_vol = recent_data.pct_change().std().mean()
        previous_vol = previous_data.pct_change().std().mean()
        
        vol_ratio = recent_vol / (previous_vol + 1e-8)
        
        # 변동성이 50% 이상 변화하면 체제 변화로 판단
        changed = vol_ratio > 1.5 or vol_ratio < 0.67
        
        return {
            'changed': changed,
            'recent_volatility': recent_vol,
            'previous_volatility': previous_vol,
            'volatility_ratio': vol_ratio,
            'regime_type': 'high_vol' if recent_vol > previous_vol else 'low_vol'
        }
    
    def _detect_correlation_regime_change(self, recent_data: pd.DataFrame,
                                        previous_data: pd.DataFrame) -> Dict:
        """
        상관관계 체제 변화 감지
        """
        recent_corr = recent_data.pct_change().corr().values
        previous_corr = previous_data.pct_change().corr().values
        
        # 상관관계 매트릭스의 평균 변화
        recent_avg_corr = recent_corr[np.triu_indices_from(recent_corr, k=1)].mean()
        previous_avg_corr = previous_corr[np.triu_indices_from(previous_corr, k=1)].mean()
        
        corr_change = abs(recent_avg_corr - previous_avg_corr)
        
        # 상관관계가 0.2 이상 변화하면 체제 변화로 판단
        changed = corr_change > 0.2
        
        return {
            'changed': changed,
            'recent_correlation': recent_avg_corr,
            'previous_correlation': previous_avg_corr,
            'correlation_change': corr_change,
            'regime_type': 'high_corr' if recent_avg_corr > previous_avg_corr else 'low_corr'
        }
    
    def _detect_trend_regime_change(self, recent_data: pd.DataFrame,
                                   previous_data: pd.DataFrame) -> Dict:
        """
        트렌드 체제 변화 감지
        """
        recent_returns = recent_data.pct_change().mean().mean()
        previous_returns = previous_data.pct_change().mean().mean()
        
        # 트렌드 방향 변화
        trend_reversal = (recent_returns > 0) != (previous_returns > 0)
        
        # 트렌드 강도 변화
        trend_strength_change = abs(recent_returns) / (abs(previous_returns) + 1e-8)
        significant_strength_change = trend_strength_change > 2.0 or trend_strength_change < 0.5
        
        changed = trend_reversal or significant_strength_change
        
        return {
            'changed': changed,
            'trend_reversal': trend_reversal,
            'recent_trend': recent_returns,
            'previous_trend': previous_returns,
            'trend_strength_change': trend_strength_change,
            'regime_type': 'uptrend' if recent_returns > 0 else 'downtrend'
        }
    
    def _classify_current_regime(self, data: pd.DataFrame) -> str:
        """
        현재 체제 분류
        """
        returns = data.pct_change().dropna()
        
        # 변동성 수준
        volatility = returns.std().mean()
        vol_level = 'high' if volatility > 0.02 else 'low'
        
        # 트렌드 방향
        trend = returns.mean().mean()
        trend_direction = 'up' if trend > 0 else 'down'
        
        # 상관관계 수준
        correlation = returns.corr().values[np.triu_indices_from(returns.corr().values, k=1)].mean()
        corr_level = 'high' if correlation > 0.5 else 'low'
        
        return f"{vol_level}vol_{trend_direction}trend_{corr_level}corr"
    
    def _calculate_change_magnitude(self, vol_change: Dict, corr_change: Dict, 
                                  trend_change: Dict) -> float:
        """
        체제 변화 크기 계산
        """
        magnitude = 0.0
        
        if vol_change['changed']:
            magnitude += abs(np.log(vol_change['volatility_ratio'])) * 0.4
        
        if corr_change['changed']:
            magnitude += corr_change['correlation_change'] * 0.3
        
        if trend_change['changed']:
            magnitude += abs(trend_change['trend_strength_change'] - 1.0) * 0.3
        
        return magnitude


class TailRiskMonitor:
    """
    테일 리스크 모니터링
    - 극단적 사건 감지
    - 테일 의존성 분석
    - 블랙 스완 이벤트 예측
    """
    
    def __init__(self):
        self.extreme_events = deque(maxlen=100)
        self.tail_threshold = 0.05  # 5% 테일
        
    def assess_tail_risk(self, market_data: pd.DataFrame) -> Dict:
        """
        테일 리스크 평가
        """
        if len(market_data) < 100:
            return {'risk_level': 0.0, 'extreme_events': [], 'tail_dependence': 0.0}
        
        returns = market_data.pct_change().dropna()
        
        # 1. 극단적 사건 감지
        extreme_events = self._detect_extreme_events(returns)
        
        # 2. 테일 의존성 분석
        tail_dependence = self._calculate_tail_dependence(returns)
        
        # 3. 테일 리스크 수준 계산
        risk_level = self._calculate_tail_risk_level(returns, extreme_events, tail_dependence)
        
        return {
            'risk_level': risk_level,
            'extreme_events': extreme_events,
            'tail_dependence': tail_dependence,
            'tail_statistics': self._calculate_tail_statistics(returns)
        }
    
    def _detect_extreme_events(self, returns: pd.DataFrame) -> List[Dict]:
        """
        극단적 사건 감지
        """
        extreme_events = []
        
        for column in returns.columns:
            series = returns[column]
            threshold = series.quantile(self.tail_threshold)
            
            extreme_indices = series[series <= threshold].index
            for idx in extreme_indices:
                event = {
                    'timestamp': idx,
                    'asset': column,
                    'return': series[idx],
                    'severity': abs(series[idx] / series.std()),
                    'percentile': (series <= series[idx]).mean()
                }
                extreme_events.append(event)
        
        # 시간순으로 정렬
        extreme_events.sort(key=lambda x: x['timestamp'])
        
        return extreme_events
    
    def _calculate_tail_dependence(self, returns: pd.DataFrame) -> float:
        """
        테일 의존성 계산
        """
        if len(returns.columns) < 2:
            return 0.0
        
        # 하부 테일 의존성 계산
        tail_dependences = []
        
        for i in range(len(returns.columns)):
            for j in range(i+1, len(returns.columns)):
                series1 = returns.iloc[:, i]
                series2 = returns.iloc[:, j]
                
                # 테일 임계값
                threshold1 = series1.quantile(self.tail_threshold)
                threshold2 = series2.quantile(self.tail_threshold)
                
                # 동시 테일 이벤트 발생 확률
                joint_tail_prob = ((series1 <= threshold1) & (series2 <= threshold2)).mean()
                marginal_tail_prob = self.tail_threshold
                
                # 테일 의존성 계수
                if marginal_tail_prob > 0:
                    tail_dependence = joint_tail_prob / marginal_tail_prob
                    tail_dependences.append(tail_dependence)
        
        return np.mean(tail_dependences) if tail_dependences else 0.0
    
    def _calculate_tail_risk_level(self, returns: pd.DataFrame, 
                                  extreme_events: List[Dict], 
                                  tail_dependence: float) -> float:
        """
        테일 리스크 수준 계산
        """
        # 최근 극단적 사건 빈도
        recent_events = [e for e in extreme_events if e['timestamp'] >= returns.index[-20]]
        event_frequency = len(recent_events) / 20.0
        
        # 극단적 사건 심각도
        if recent_events:
            avg_severity = np.mean([e['severity'] for e in recent_events])
        else:
            avg_severity = 0.0
        
        # 테일 의존성 위험
        dependence_risk = tail_dependence * 0.5
        
        # 전체 테일 리스크 수준
        risk_level = (event_frequency * 0.4 + avg_severity * 0.4 + dependence_risk * 0.2)
        
        return np.clip(risk_level, 0.0, 1.0)
    
    def _calculate_tail_statistics(self, returns: pd.DataFrame) -> Dict:
        """
        테일 통계 계산
        """
        tail_stats = {}
        
        for column in returns.columns:
            series = returns[column]
            
            # 테일 통계
            var_95 = series.quantile(0.05)
            var_99 = series.quantile(0.01)
            
            # 조건부 VaR (Expected Shortfall)
            es_95 = series[series <= var_95].mean()
            es_99 = series[series <= var_99].mean()
            
            tail_stats[column] = {
                'var_95': var_95,
                'var_99': var_99,
                'es_95': es_95,
                'es_99': es_99,
                'tail_index': self._calculate_tail_index(series)
            }
        
        return tail_stats
    
    def _calculate_tail_index(self, series: pd.Series) -> float:
        """
        테일 인덱스 계산 (Hill estimator)
        """
        # 절댓값 순서 통계량
        abs_series = abs(series).sort_values(ascending=False)
        
        # 상위 10% 사용
        k = int(len(abs_series) * 0.1)
        if k < 10:
            return 1.0
        
        # Hill estimator
        log_ratios = np.log(abs_series.iloc[:k] / abs_series.iloc[k])
        tail_index = np.mean(log_ratios)
        
        return tail_index


class AdaptivePositionSizer:
    """
    적응적 포지션 사이징
    - 리스크 기반 포지션 조정
    - 켈리 기준 응용
    - 동적 레버리지 조정
    """
    
    def __init__(self, max_position_size: float = 0.2, 
                 max_portfolio_risk: float = 0.15):
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.position_history = deque(maxlen=100)
        
    def calculate_position_sizes(self, 
                               expected_returns: np.ndarray,
                               covariance_matrix: np.ndarray,
                               crisis_level: float,
                               current_positions: np.ndarray) -> np.ndarray:
        """
        적응적 포지션 사이징
        """
        n_assets = len(expected_returns)
        
        # 1. 기본 켈리 기준 포지션
        kelly_positions = self._calculate_kelly_positions(
            expected_returns, covariance_matrix
        )
        
        # 2. 위기 수준에 따른 조정
        crisis_adjustment = self._calculate_crisis_adjustment(crisis_level)
        adjusted_positions = kelly_positions * crisis_adjustment
        
        # 3. 리스크 예산 제약
        risk_adjusted_positions = self._apply_risk_budget_constraints(
            adjusted_positions, covariance_matrix
        )
        
        # 4. 거래 비용 고려
        transaction_adjusted_positions = self._apply_transaction_cost_adjustment(
            risk_adjusted_positions, current_positions
        )
        
        # 5. 포지션 크기 제한
        final_positions = self._apply_position_limits(transaction_adjusted_positions)
        
        # 포지션 히스토리 업데이트
        self.position_history.append({
            'kelly_positions': kelly_positions,
            'crisis_adjustment': crisis_adjustment,
            'final_positions': final_positions,
            'crisis_level': crisis_level
        })
        
        return final_positions
    
    def _calculate_kelly_positions(self, expected_returns: np.ndarray,
                                 covariance_matrix: np.ndarray) -> np.ndarray:
        """
        켈리 기준 포지션 계산
        """
        try:
            # 켈리 공식: f = (μ - r) / σ²
            # 여기서 r은 무위험 수익률 (0으로 가정)
            
            # 공분산 행렬의 역행렬
            inv_cov = np.linalg.inv(covariance_matrix + np.eye(len(covariance_matrix)) * 1e-8)
            
            # 켈리 포지션
            kelly_positions = inv_cov @ expected_returns
            
            # 정규화 (총 포지션이 1을 초과하지 않도록)
            total_abs_position = np.sum(np.abs(kelly_positions))
            if total_abs_position > 1.0:
                kelly_positions = kelly_positions / total_abs_position
            
            return kelly_positions
            
        except np.linalg.LinAlgError:
            # 공분산 행렬이 특이행렬인 경우 균등 가중
            return np.ones(len(expected_returns)) / len(expected_returns)
    
    def _calculate_crisis_adjustment(self, crisis_level: float) -> float:
        """
        위기 수준에 따른 포지션 조정
        """
        # 위기 수준이 높을수록 포지션 감소
        if crisis_level < 0.3:
            return 1.0  # 정상 시장
        elif crisis_level < 0.6:
            return 0.7  # 중간 위기
        elif crisis_level < 0.8:
            return 0.4  # 높은 위기
        else:
            return 0.2  # 극심한 위기
    
    def _apply_risk_budget_constraints(self, positions: np.ndarray,
                                     covariance_matrix: np.ndarray) -> np.ndarray:
        """
        리스크 예산 제약 적용
        """
        # 포트폴리오 변동성 계산
        portfolio_variance = positions.T @ covariance_matrix @ positions
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # 리스크 예산 초과시 축소
        if portfolio_volatility > self.max_portfolio_risk:
            scaling_factor = self.max_portfolio_risk / portfolio_volatility
            positions = positions * scaling_factor
        
        return positions
    
    def _apply_transaction_cost_adjustment(self, target_positions: np.ndarray,
                                         current_positions: np.ndarray) -> np.ndarray:
        """
        거래 비용 고려 조정
        """
        # 거래 비용 (간단한 모델)
        transaction_cost_rate = 0.001  # 0.1%
        
        # 포지션 변화량
        position_changes = target_positions - current_positions
        
        # 거래 비용이 예상 이익보다 큰 경우 거래 안함
        adjusted_positions = current_positions.copy()
        
        for i in range(len(position_changes)):
            change = position_changes[i]
            cost = abs(change) * transaction_cost_rate
            
            # 변화량이 거래 비용의 5배 이상인 경우만 거래
            if abs(change) > cost * 5:
                adjusted_positions[i] = target_positions[i]
        
        return adjusted_positions
    
    def _apply_position_limits(self, positions: np.ndarray) -> np.ndarray:
        """
        포지션 크기 제한 적용
        """
        # 개별 자산 포지션 제한
        limited_positions = np.clip(positions, -self.max_position_size, self.max_position_size)
        
        # 총 포지션 제한
        total_abs_position = np.sum(np.abs(limited_positions))
        if total_abs_position > 1.0:
            limited_positions = limited_positions / total_abs_position
        
        return limited_positions


class BIPDRiskManager:
    """
    BIPD 통합 리스크 관리 시스템
    """
    
    def __init__(self, lookback_periods: List[int] = [5, 20, 60]):
        self.crisis_detector = CrisisDetectionSystem(lookback_periods)
        self.position_sizer = AdaptivePositionSizer()
        
        # 리스크 지표 추적
        self.risk_metrics_history = deque(maxlen=1000)
        
    def assess_and_manage_risk(self, 
                              market_data: pd.DataFrame,
                              current_features: np.ndarray,
                              expected_returns: np.ndarray,
                              current_positions: np.ndarray) -> Dict:
        """
        종합 리스크 평가 및 관리
        """
        # 1. 위기 감지
        crisis_info = self.crisis_detector.detect_crisis(market_data, current_features)
        
        # 2. 공분산 행렬 계산
        returns = market_data.pct_change().dropna()
        covariance_matrix = returns.cov().values
        
        # 3. 적응적 포지션 사이징
        recommended_positions = self.position_sizer.calculate_position_sizes(
            expected_returns, covariance_matrix, 
            crisis_info['overall_crisis_level'], current_positions
        )
        
        # 4. 리스크 지표 계산
        risk_metrics = self._calculate_comprehensive_risk_metrics(
            market_data, current_positions, recommended_positions
        )
        
        # 5. 리스크 관리 권고사항 생성
        recommendations = self._generate_risk_recommendations(
            crisis_info, risk_metrics, current_positions, recommended_positions
        )
        
        # 히스토리 업데이트
        self.risk_metrics_history.append({
            'crisis_info': crisis_info,
            'risk_metrics': risk_metrics,
            'current_positions': current_positions,
            'recommended_positions': recommended_positions,
            'recommendations': recommendations
        })
        
        return {
            'crisis_info': crisis_info,
            'risk_metrics': risk_metrics,
            'current_positions': current_positions,
            'recommended_positions': recommended_positions,
            'recommendations': recommendations,
            'risk_dashboard': self._create_risk_dashboard()
        }
    
    def _calculate_comprehensive_risk_metrics(self, 
                                            market_data: pd.DataFrame,
                                            current_positions: np.ndarray,
                                            recommended_positions: np.ndarray) -> Dict:
        """
        종합 리스크 지표 계산
        """
        returns = market_data.pct_change().dropna()
        
        # 현재 포트폴리오 리스크
        current_portfolio_return = (returns * current_positions).sum(axis=1)
        current_var_95 = current_portfolio_return.quantile(0.05)
        current_volatility = current_portfolio_return.std()
        
        # 권고 포트폴리오 리스크
        recommended_portfolio_return = (returns * recommended_positions).sum(axis=1)
        recommended_var_95 = recommended_portfolio_return.quantile(0.05)
        recommended_volatility = recommended_portfolio_return.std()
        
        # 리스크 기여도 분석
        risk_contributions = self._calculate_risk_contributions(
            returns, recommended_positions
        )
        
        # 스트레스 테스트
        stress_test_results = self._conduct_stress_tests(
            returns, recommended_positions
        )
        
        return {
            'current_portfolio': {
                'var_95': current_var_95,
                'volatility': current_volatility,
                'sharpe_ratio': current_portfolio_return.mean() / (current_volatility + 1e-8)
            },
            'recommended_portfolio': {
                'var_95': recommended_var_95,
                'volatility': recommended_volatility,
                'sharpe_ratio': recommended_portfolio_return.mean() / (recommended_volatility + 1e-8)
            },
            'risk_contributions': risk_contributions,
            'stress_test_results': stress_test_results,
            'risk_improvement': {
                'var_improvement': (current_var_95 - recommended_var_95) / abs(current_var_95 + 1e-8),
                'volatility_improvement': (current_volatility - recommended_volatility) / (current_volatility + 1e-8)
            }
        }
    
    def _calculate_risk_contributions(self, returns: pd.DataFrame, 
                                    positions: np.ndarray) -> Dict:
        """
        리스크 기여도 분석
        """
        # 포트폴리오 수익률
        portfolio_returns = (returns * positions).sum(axis=1)
        portfolio_variance = portfolio_returns.var()
        
        # 각 자산의 리스크 기여도
        contributions = {}
        for i, asset in enumerate(returns.columns):
            # marginal VaR 계산
            asset_contribution = positions[i] * returns.iloc[:, i].cov(portfolio_returns)
            contribution_percentage = asset_contribution / portfolio_variance
            
            contributions[asset] = {
                'absolute_contribution': asset_contribution,
                'percentage_contribution': contribution_percentage,
                'position_size': positions[i]
            }
        
        return contributions
    
    def _conduct_stress_tests(self, returns: pd.DataFrame, 
                            positions: np.ndarray) -> Dict:
        """
        스트레스 테스트 수행
        """
        portfolio_returns = (returns * positions).sum(axis=1)
        
        # 시나리오 기반 스트레스 테스트
        scenarios = {
            'market_crash': -0.20,  # 20% 하락
            'high_volatility': returns.std() * 3,  # 변동성 3배 증가
            'correlation_spike': 0.9,  # 상관관계 0.9로 증가
            'black_swan': -0.10  # 10% 극단적 손실
        }
        
        stress_results = {}
        for scenario, shock in scenarios.items():
            if scenario == 'market_crash':
                stressed_returns = returns + shock
            elif scenario == 'high_volatility':
                stressed_returns = returns * 3
            elif scenario == 'correlation_spike':
                # 상관관계 증가 시뮬레이션 (단순화)
                mean_return = returns.mean()
                stressed_returns = returns * 0.5 + mean_return * 0.5
            else:  # black_swan
                stressed_returns = returns.copy()
                stressed_returns.iloc[-1] = shock
            
            stressed_portfolio = (stressed_returns * positions).sum(axis=1)
            
            stress_results[scenario] = {
                'portfolio_return': stressed_portfolio.iloc[-1],
                'var_95': stressed_portfolio.quantile(0.05),
                'max_loss': stressed_portfolio.min(),
                'volatility': stressed_portfolio.std()
            }
        
        return stress_results
    
    def _generate_risk_recommendations(self, 
                                     crisis_info: Dict,
                                     risk_metrics: Dict,
                                     current_positions: np.ndarray,
                                     recommended_positions: np.ndarray) -> List[str]:
        """
        리스크 관리 권고사항 생성
        """
        recommendations = []
        
        # 위기 수준 기반 권고
        crisis_level = crisis_info['overall_crisis_level']
        if crisis_level > 0.8:
            recommendations.append("CRITICAL: Extreme crisis detected. Consider significant position reduction.")
        elif crisis_level > 0.6:
            recommendations.append("HIGH RISK: Elevated crisis level. Reduce position sizes and increase hedging.")
        elif crisis_level > 0.4:
            recommendations.append("MEDIUM RISK: Moderate crisis signals. Monitor closely and consider minor adjustments.")
        
        # 포지션 변화 권고
        position_changes = recommended_positions - current_positions
        significant_changes = np.abs(position_changes) > 0.05
        
        if np.any(significant_changes):
            recommendations.append("Position rebalancing recommended based on current risk assessment.")
        
        # 리스크 지표 기반 권고
        if risk_metrics['current_portfolio']['var_95'] < -0.05:
            recommendations.append("Current portfolio shows high downside risk. Consider risk reduction measures.")
        
        if risk_metrics['risk_improvement']['var_improvement'] > 0.1:
            recommendations.append("Recommended positions show significant risk improvement opportunity.")
        
        # 스트레스 테스트 기반 권고
        stress_results = risk_metrics['stress_test_results']
        if stress_results['market_crash']['portfolio_return'] < -0.15:
            recommendations.append("Portfolio vulnerable to market crash scenarios. Increase diversification.")
        
        if stress_results['high_volatility']['volatility'] > 0.3:
            recommendations.append("High sensitivity to volatility spikes. Consider volatility hedging.")
        
        return recommendations
    
    def _create_risk_dashboard(self) -> Dict:
        """
        리스크 대시보드 생성
        """
        if len(self.risk_metrics_history) < 10:
            return {'status': 'insufficient_data'}
        
        recent_history = list(self.risk_metrics_history)[-10:]
        
        # 리스크 추세 분석
        crisis_levels = [h['crisis_info']['overall_crisis_level'] for h in recent_history]
        volatilities = [h['risk_metrics']['current_portfolio']['volatility'] for h in recent_history]
        
        dashboard = {
            'crisis_trend': {
                'current_level': crisis_levels[-1],
                'trend': np.polyfit(range(len(crisis_levels)), crisis_levels, 1)[0],
                'volatility_trend': np.polyfit(range(len(volatilities)), volatilities, 1)[0]
            },
            'risk_alerts': self._generate_risk_alerts(recent_history),
            'performance_summary': {
                'avg_crisis_level': np.mean(crisis_levels),
                'avg_volatility': np.mean(volatilities),
                'risk_stability': 1.0 - np.std(crisis_levels)
            }
        }
        
        return dashboard
    
    def _generate_risk_alerts(self, recent_history: List[Dict]) -> List[str]:
        """
        리스크 알림 생성
        """
        alerts = []
        
        # 위기 수준 급증 알림
        crisis_levels = [h['crisis_info']['overall_crisis_level'] for h in recent_history]
        if len(crisis_levels) >= 2 and crisis_levels[-1] - crisis_levels[-2] > 0.3:
            alerts.append("ALERT: Rapid increase in crisis level detected!")
        
        # 변동성 급증 알림
        volatilities = [h['risk_metrics']['current_portfolio']['volatility'] for h in recent_history]
        if len(volatilities) >= 2 and volatilities[-1] / volatilities[-2] > 1.5:
            alerts.append("ALERT: Significant volatility spike detected!")
        
        # 체제 변화 알림
        if recent_history[-1]['crisis_info']['regime_change']['regime_changed']:
            alerts.append("ALERT: Market regime change detected!")
        
        return alerts