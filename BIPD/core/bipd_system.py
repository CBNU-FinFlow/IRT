"""
BIPD 통합 시스템
생체면역 기반 포트폴리오 방어 시스템의 핵심 구현
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings
from tqdm import tqdm
import time

from agents.tcell import TCell
from agents.bcell import BCell
from memory.memory_cell import MemorySystem
from risk.risk_management import RiskManager, Calculations
from utils.data_loader import (
    download_market_data, process_market_data, calculate_technical_indicators,
    calculate_returns, extract_market_features, 
    get_default_symbols, clean_data
)
from temporal.temporal_mechanisms import (
    AdaptiveTimeWindowManager, CyclicalPatternDetector, TemporalStateEncoder
)

warnings.filterwarnings("ignore")


class BIPDSystem:
    """
    생체면역 기반 포트폴리오 방어 시스템
    T-Cell, B-Cell, Memory Cell을 통합한 적응형 포트폴리오 관리
    """
    
    def __init__(self, 
                 symbols: List[str] = None,
                 initial_capital: float = 100000,
                 lookback_window: int = 252,
                 rebalance_frequency: int = 5):
        
        # 기본 설정
        self.symbols = symbols or get_default_symbols()
        self.n_assets = len(self.symbols)
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        self.rebalance_frequency = rebalance_frequency
        
        # 시스템 컴포넌트 초기화
        self._initialize_immune_system()
        self._initialize_support_systems()
        
        # 상태 변수
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = initial_capital
        self.step_count = 0
        
        # 히스토리 추적
        self.performance_history = []
        self.weights_history = []
        self.crisis_history = []
        self.decision_history = []
        
        # 데이터 캐시
        self.price_data = None
        self.market_features = None
        self.is_fitted = False
    
    def _initialize_immune_system(self):
        """면역 시스템 초기화"""
        # T-Cell 위기 감지
        self.tcell = TCell(
            cell_id="Crisis-Detector",
            sensitivity=0.05
        )
        
        # B-Cell 전문화 전략
        self.bcells = {
            'volatility': BCell(
                cell_id="Volatility-Specialist",
                specialty_type="volatility",
                n_assets=self.n_assets
            ),
            'correlation': BCell(
                cell_id="Correlation-Specialist", 
                specialty_type="correlation",
                n_assets=self.n_assets
            ),
            'momentum': BCell(
                cell_id="Momentum-Specialist",
                specialty_type="momentum", 
                n_assets=self.n_assets
            ),
            'liquidity': BCell(
                cell_id="Liquidity-Specialist",
                specialty_type="liquidity",
                n_assets=self.n_assets
            ),
            'memory_recall': BCell(
                cell_id="Memory-Specialist",
                specialty_type="memory_recall",
                n_assets=self.n_assets
            )
        }
        
        # 메모리 시스템
        self.memory_system = MemorySystem(n_memory_cells=3)
    
    def _initialize_support_systems(self):
        """지원 시스템 초기화"""
        # 리스크 관리
        self.risk_manager = RiskManager(
            max_position_size=0.4,
            max_turnover=0.3,
            min_position_size=0.01,
            max_concentration=0.6
        )
        
        # 시간적 메커니즘
        self.time_window_manager = AdaptiveTimeWindowManager()
        self.pattern_detector = CyclicalPatternDetector()
        self.state_encoder = TemporalStateEncoder(input_dim=12)  # 기본 시장 특성 차원
        
        # 계산 유틸리티
        self.calc = Calculations()
        
    def extract_market_features(self, market_data, lookback=20):
        """시장 특성 추출"""
        if len(market_data) < lookback:
            return np.zeros(12)
        
        returns = market_data.pct_change().dropna()
        if len(returns) == 0:
            return np.zeros(12)
        
        recent_returns = returns.iloc[-lookback:]
        if len(recent_returns) == 0:
            return np.zeros(12)
            
        # 기본 특성 계산
        daily_return = self.calc.mean(recent_returns.mean(axis=1))
        volatility = self.calc.std(recent_returns.std(axis=1))
        
        # 상관관계
        correlation = self.calc.corr(recent_returns)
        
        # 모멘텀
        momentum = self.calc.momentum(recent_returns)
        
        # 유동성 (거래량 기반 추정)
        liquidity = 1.0 - min(volatility / 0.05, 1.0)
        
        # 시장 스트레스
        market_stress = min(volatility / 0.03, 1.0)
        
        # 추가 특성
        max_return = self.calc.mean(recent_returns.max(axis=1))
        min_return = self.calc.mean(recent_returns.min(axis=1))
        
        # 변동성 클러스터링
        vol_clustering = self.calc.std(recent_returns.std(axis=1))
        
        # 왜도와 첨도
        skewness = self.calc.skew(recent_returns.mean(axis=1))
        kurtosis = self.calc.kurtosis(recent_returns.mean(axis=1))
        
        # 트렌드
        trend = self.calc.trend(recent_returns.mean(axis=1))
        
        return np.array([
            daily_return, volatility, correlation, momentum, liquidity,
            market_stress, max_return, min_return, vol_clustering,
            skewness, kurtosis, trend
        ])
        
    def immune_response(self, market_features, training=False):
        """
        면역 반응 실행
        
        Returns:
            weights: 포트폴리오 가중치
            response_type: 반응 유형
            crisis_level: 위기 수준
        """
        # 1. T-Cell 위기 감지
        try:
            crisis_result = self.tcell.detect_crisis(market_features)
            crisis_level = crisis_result.get('crisis_level', 0.0)
        except:
            crisis_level = 0.0
            
        # 2. B-Cell 전략 생성
        try:
            bcell_weights = []
            for specialty, bcell in self.bcells.items():
                weight = bcell.generate_strategy(market_features)
                bcell_weights.append(weight)
            
            # 가중 평균 계산
            if bcell_weights:
                weights = np.mean(bcell_weights, axis=0)
                weights = np.abs(weights)  # 양수 보장
                weights = weights / np.sum(weights)  # 정규화
            else:
                weights = np.ones(self.n_assets) / self.n_assets
        except:
            weights = np.ones(self.n_assets) / self.n_assets
            
        # 3. 위기 수준에 따른 조정
        if crisis_level > 0.7:
            # 고위기: 보수적 전략
            conservative_weights = np.ones(self.n_assets) / self.n_assets * 0.5
            weights = 0.3 * weights + 0.7 * conservative_weights
            response_type = "defensive"
        elif crisis_level > 0.4:
            # 중위기: 균형 전략
            response_type = "balanced"
        else:
            # 저위기: 공격적 전략
            response_type = "aggressive"
            
        # 4. 최종 정규화
        weights = np.abs(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(self.n_assets) / self.n_assets
            
        return weights, response_type, crisis_level
        
    def pretrain_bcells(self, market_data, episodes=500):
        """B-Cell 사전 훈련"""
        # 전문 정책 함수들
        expert_policies = {
            "volatility": self._volatility_response,
            "correlation": self._correlation_response, 
            "momentum": self._momentum_response,
            "liquidity": self._liquidity_response,
            "memory_recall": self._memory_response
        }
        
        # tqdm으로 진행률 표시
        with tqdm(total=episodes, desc="B-Cell 사전 훈련", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            
            for episode in range(episodes):
                # 랜덤 시점 선택
                start_idx = np.random.randint(20, len(market_data) - 50)
                current_data = market_data.iloc[:start_idx]
                market_features = self.extract_market_features(current_data)
                crisis_level = np.random.uniform(0.2, 0.8)
                
                # 각 B-Cell 훈련
                for specialty, bcell in self.bcells.items():
                    if specialty in expert_policies:
                        # 전문 정책으로부터 타겟 액션 생성
                        expert_action = expert_policies[specialty](crisis_level)
                        
                        # B-Cell 네트워크 훈련
                        bcell.train_step(market_features, expert_action)
                
                pbar.update(1)
        
    def _volatility_response(self, crisis_level):
        """변동성 전문 정책"""
        base_weights = np.ones(self.n_assets) / self.n_assets
        volatility_factor = 1.0 - crisis_level * 0.5
        return base_weights * volatility_factor
        
    def _correlation_response(self, crisis_level):
        """상관관계 전문 정책"""
        base_weights = np.ones(self.n_assets) / self.n_assets
        if crisis_level > 0.6:
            # 고위기시 분산 투자
            return base_weights * 0.8
        return base_weights
        
    def _momentum_response(self, crisis_level):
        """모멘텀 전문 정책"""
        base_weights = np.ones(self.n_assets) / self.n_assets
        momentum_factor = 1.0 + (1.0 - crisis_level) * 0.3
        return base_weights * momentum_factor
        
    def _liquidity_response(self, crisis_level):
        """유동성 전문 정책"""
        base_weights = np.ones(self.n_assets) / self.n_assets
        liquidity_factor = 1.0 - crisis_level * 0.3
        return base_weights * liquidity_factor
        
    def _memory_response(self, crisis_level):
        """메모리 기반 정책"""
        base_weights = np.ones(self.n_assets) / self.n_assets
        memory_factor = 1.0 - crisis_level * 0.2
        return base_weights * memory_factor
    
    def load_data(self, 
                  start_date: str = "2008-01-01",
                  end_date: str = "2024-12-31"):
        """
        시장 데이터 로드 및 전처리
        
        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
        """
        print(f"데이터 로딩 중: {self.symbols}")
        
        # 데이터 다운로드
        market_data = download_market_data(
            symbols=self.symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        # 데이터 구조 분해
        self.price_data = market_data["prices"]
        self.technical_indicators = market_data["features"]
        self.raw_data = market_data["raw_data"]
        
        # 수익률 계산
        self.returns_data = calculate_returns(self.price_data)
        
        # 시장 특성 추출
        self.market_features = self.technical_indicators
        
        # 데이터 정리
        self.price_data = clean_data(self.price_data)
        self.returns_data = clean_data(self.returns_data)
        self.market_features = clean_data(self.market_features)
        
        print(f"데이터 로딩 완료:")
        print(f"- 가격 데이터: {self.price_data.shape}")
        print(f"- 기술적 지표: {self.technical_indicators.shape}")
        print(f"- 시장 특성: {self.market_features.shape}")
        
        return self
    
    def fit(self, train_end_ratio: float = 0.8):
        """
        시스템 훈련
        
        Args:
            train_end_ratio: 훈련 데이터 비율
        """
        if self.price_data is None:
            raise ValueError("먼저 데이터를 로드하세요 (load_data)")
        
        # 훈련 데이터 분할
        train_size = int(len(self.price_data) * train_end_ratio)
        train_features = self.market_features.iloc[:train_size]
        
        # T-Cell 훈련
        with tqdm(total=1, desc="T-Cell Training", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            self.tcell.fit(train_features.values)
            pbar.update(1)
        
        # B-Cell 사전 훈련
        self.pretrain_bcells(self.price_data.iloc[:train_size], episodes=500)
        
        # 시간적 패턴 학습
        with tqdm(total=1, desc="Temporal Pattern Learning", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            self.pattern_detector.fit(train_features.values)
            self.time_window_manager.initialize(train_features.values)
            pbar.update(1)
        
        # 실제 강화학습 수행
        self._run_reinforcement_learning(self.price_data.iloc[:train_size], episodes=100)
        
        self.is_fitted = True
        print("시스템 훈련 완료")
        return self
    
    def _run_reinforcement_learning(self, train_data: pd.DataFrame, episodes: int = 100):
        """실제 강화학습 수행"""
        
        with tqdm(total=episodes, desc="Reinforcement Learning Episodes", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            
            for episode in range(episodes):
                # 랜덤 시작점 선택
                start_idx = np.random.randint(50, len(train_data) - 100)
                episode_length = min(50, len(train_data) - start_idx)
                
                episode_rewards = []
                
                for step in range(episode_length):
                    current_idx = start_idx + step
                    
                    # 현재까지의 데이터로 시장 특성 추출
                    current_data = train_data.iloc[:current_idx+1]
                    market_features = self.extract_market_features(current_data)
                    
                    # Temporal 분석 추가
                    current_volatility = self.calc.std(current_data.pct_change().dropna().std(axis=1))
                    adaptive_window = self.time_window_manager.calculate_adaptive_window(
                        current_data, current_volatility, 0.0
                    )
                    cycle_info = self.pattern_detector.detect_cycles(current_data)
                    
                    # 면역 반응 실행
                    weights, response_type, crisis_level = self.immune_response(
                        market_features, training=True
                    )
                    
                    # 다음 스텝의 실제 수익률 계산
                    if current_idx + 1 < len(train_data):
                        next_returns = train_data.iloc[current_idx+1].pct_change()
                        portfolio_return = np.sum(weights * next_returns.fillna(0))
                        episode_rewards.append(portfolio_return)
                        
                        # 경험 저장
                        self._store_experience(
                            market_features, weights, portfolio_return, crisis_level
                        )
                        
                        # B-Cell 학습
                        self._update_bcells(
                            market_features, weights, portfolio_return, crisis_level
                        )
                
                pbar.update(1)
    
    def step(self, current_idx: int) -> Dict[str, Any]:
        """
        단일 시점 실행
        
        Args:
            current_idx: 현재 시점 인덱스
            
        Returns:
            실행 결과 딕셔너리
        """
        if not self.is_fitted:
            raise ValueError("시스템이 훈련되지 않았습니다 (fit)")
        
        # 현재 시장 상태 획득
        current_features = self.market_features.iloc[current_idx].values
        current_prices = self.price_data.iloc[current_idx].values
        
        # 1. T-Cell 위기 감지
        crisis_result = self.tcell.detect_crisis(current_features)
        crisis_level = crisis_result['crisis_level']
        
        # 2. 메모리 시스템 회상
        memory_result = self.memory_system.recall_experience(
            current_features, crisis_level
        )
        
        # 3. B-Cell 전략 생성
        bcell_weights = {}
        bcell_explanations = {}
        
        for specialty, bcell in self.bcells.items():
            weights = bcell.generate_portfolio_weights(current_features, crisis_level)
            explanation = bcell.get_decision_explanation(current_features, crisis_level)
            
            bcell_weights[specialty] = weights
            bcell_explanations[specialty] = explanation
        
        # 4. 통합 포트폴리오 생성
        integrated_weights = self._integrate_strategies(
            bcell_weights, crisis_level, memory_result
        )
        
        # 5. 리스크 관리 적용
        final_weights = self.risk_manager.apply_constraints(
            integrated_weights, current_prices, crisis_level
        )
        
        # 6. 성과 계산
        if self.step_count > 0:
            period_return = self._calculate_period_return(
                current_prices, final_weights
            )
            
            # 경험 저장
            self._store_experience(
                current_features, final_weights, period_return, crisis_level
            )
            
            # B-Cell 학습
            self._update_bcells(
                current_features, final_weights, period_return, crisis_level
            )
        
        # 7. 포트폴리오 업데이트
        self.current_weights = final_weights
        self.step_count += 1
        
        # 8. 결과 기록
        step_result = {
            'timestamp': self.price_data.index[current_idx],
            'crisis_detection': crisis_result,
            'memory_recall': memory_result,
            'bcell_explanations': bcell_explanations,
            'integrated_weights': integrated_weights,
            'final_weights': final_weights,
            'crisis_level': crisis_level,
            'portfolio_value': self.portfolio_value
        }
        
        self._update_history(step_result)
        
        return step_result
    
    def _integrate_strategies(self, 
                            bcell_weights: Dict[str, np.ndarray],
                            crisis_level: float,
                            memory_result: Optional[Dict[str, Any]]) -> np.ndarray:
        """전략 통합"""
        # 기본 균등 가중치
        integrated_weights = np.zeros(self.n_assets)
        total_weight = 0.0
        
        # B-Cell 전략 통합
        for specialty, bcell in self.bcells.items():
            activation_strength = bcell.activation_level
            
            if activation_strength > 0:
                weight = activation_strength
                integrated_weights += weight * bcell_weights[specialty]
                total_weight += weight
        
        # 메모리 전략 통합
        if memory_result and memory_result['similarity'] > 0.8:
            memory_weight = memory_result['similarity'] * 0.3
            memory_portfolio = memory_result['memory']['portfolio_weights']
            
            # 길이 맞춤
            if len(memory_portfolio) == self.n_assets:
                integrated_weights += memory_weight * memory_portfolio
                total_weight += memory_weight
        
        # 정규화
        if total_weight > 0:
            integrated_weights /= total_weight
        else:
            integrated_weights = np.ones(self.n_assets) / self.n_assets
        
        return integrated_weights
    
    def _calculate_period_return(self, 
                               current_prices: np.ndarray,
                               weights: np.ndarray) -> float:
        """기간 수익률 계산"""
        if len(self.performance_history) == 0:
            return 0.0
        
        # 이전 가격 정보가 필요하므로 간단히 랜덤 워크 가정
        prev_idx = max(0, len(self.performance_history) - 1)
        
        if prev_idx < len(self.price_data) - 1:
            prev_prices = self.price_data.iloc[prev_idx].values
            price_changes = (current_prices - prev_prices) / prev_prices
            portfolio_return = np.dot(weights, price_changes)
            
            # 포트폴리오 가치 업데이트
            self.portfolio_value *= (1 + portfolio_return)
            
            return portfolio_return
        
        return 0.0
    
    def _store_experience(self, 
                         market_features: np.ndarray,
                         portfolio_weights: np.ndarray,
                         performance: float,
                         crisis_level: float):
        """경험 저장"""
        # 메모리 시스템에 저장
        self.memory_system.store_experience(
            market_features, portfolio_weights, performance, crisis_level
        )
    
    def _update_bcells(self, 
                      market_features: np.ndarray,
                      action: np.ndarray,
                      reward: float,
                      crisis_level: float):
        """B-Cell 학습"""
        # 다음 상태 (현재는 현재 상태와 동일하게 처리)
        next_features = market_features
        
        # 각 B-Cell 업데이트
        for bcell in self.bcells.values():
            bcell.store_experience(
                market_features, action, reward, next_features, crisis_level
            )
            
            # 배치 학습
            if self.step_count % 10 == 0:  # 10스텝마다 학습
                bcell.learn()
    
    def _update_history(self, step_result: Dict[str, Any]):
        """히스토리 업데이트"""
        self.performance_history.append(step_result['portfolio_value'])
        self.weights_history.append(step_result['final_weights'])
        self.crisis_history.append(step_result['crisis_level'])
        self.decision_history.append(step_result)
    
    def run_backtest(self, 
                    start_idx: int = None,
                    end_idx: int = None,
                    verbose: bool = True) -> Dict[str, Any]:
        """
        백테스트 실행
        
        Args:
            start_idx: 시작 인덱스
            end_idx: 종료 인덱스
            verbose: 진행상황 출력
            
        Returns:
            백테스트 결과
        """
        if not self.is_fitted:
            raise ValueError("시스템이 훈련되지 않았습니다")
        
        # 인덱스 설정
        if start_idx is None:
            start_idx = int(len(self.price_data) * 0.8)  # 훈련 데이터 이후
        if end_idx is None:
            end_idx = len(self.price_data)
        
        print(f"백테스트 시작: {start_idx} ~ {end_idx}")
        
        # 초기화
        self.portfolio_value = self.initial_capital
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.step_count = 0
        
        # 히스토리 초기화
        self.performance_history = []
        self.weights_history = []
        self.crisis_history = []
        self.decision_history = []
        
        # 백테스트 실행
        test_data = self.price_data.iloc[start_idx:end_idx]
        test_returns = test_data.pct_change().dropna()
        
        if len(test_returns) == 0:
            print("Warning: No test returns data available")
            return self._generate_backtest_results(start_idx, end_idx)
        
        portfolio_values = [self.portfolio_value]
        
        with tqdm(total=len(test_returns), desc="Backtest Execution", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            
            for i in range(len(test_returns)):
                try:
                    # 현재까지의 데이터
                    current_data = test_data.iloc[:i+1]
                    current_idx = start_idx + i
                    
                    # 시장 특성 추출 (훈련 시와 동일한 특성 사용)
                    if current_idx < len(self.market_features):
                        market_features = self.market_features.iloc[current_idx].values
                    else:
                        # 인덱스 범위를 벗어난 경우 마지막 특성 사용
                        market_features = self.market_features.iloc[-1].values
                    
                    # Temporal 분석 수행
                    current_volatility = self.calc.std(current_data.pct_change().dropna().std(axis=1))
                    adaptive_window = self.time_window_manager.calculate_adaptive_window(
                        current_data, current_volatility, 0.0
                    )
                    cycle_info = self.pattern_detector.detect_cycles(current_data)
                    
                    # 면역 반응 실행
                    weights, response_type, crisis_level = self.immune_response(
                        market_features, training=False
                    )
                    
                    # T-Cell 분석 결과 생성
                    crisis_detection = self.tcell.detect_crisis(market_features)
                    
                    # B-Cell 설명 생성
                    bcell_explanations = {}
                    for specialty, bcell in self.bcells.items():
                        bcell_explanations[specialty] = bcell.get_decision_explanation(
                            market_features, crisis_level
                        )
                    
                    # 포트폴리오 수익률 계산
                    portfolio_return = np.sum(weights * test_returns.iloc[i])
                    portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
                    
                    # 현재 포트폴리오 가치 업데이트
                    self.portfolio_value = portfolio_values[-1]
                    
                    # 기록
                    step_result = {
                        'timestamp': test_returns.index[i],
                        'market_features': market_features,
                        'crisis_detection': crisis_detection,
                        'memory_recall': None,  # 메모리 시스템 결과 추가
                        'bcell_explanations': bcell_explanations,
                        'crisis_level': crisis_level,
                        'final_weights': weights,
                        'portfolio_value': portfolio_values[-1],
                        'portfolio_return': portfolio_return,
                        'adaptive_window': adaptive_window,
                        'cycle_info': cycle_info
                    }
                    
                    self._update_history(step_result)
                    
                    # 리밸런싱 로깅 (진행률 바에 포함)
                    if i % self.rebalance_frequency == 0:
                        pbar.set_postfix({
                            'Crisis': f'{crisis_level:.3f}', 
                            'Value': f'{portfolio_values[-1]:.0f}',
                            'Return': f'{portfolio_return:.3f}'
                        })
                    
                except Exception as e:
                    print(f"\nStep {i} error: {str(e)}")
                    pbar.set_postfix({'Error': f'Step {i}'})
                    continue
                
                pbar.update(1)
        
        # 결과 생성
        results = self._generate_backtest_results(start_idx, end_idx)
        
        if 'total_return' in results:
            print(f"백테스트 완료. 총 수익률: {results['total_return']:.2%}")
        else:
            print("백테스트 완료. 결과 처리 중 오류 발생")
        
        return results
    
    def _generate_backtest_results(self, start_idx: int, end_idx: int) -> Dict[str, Any]:
        """백테스트 결과 생성"""
        if len(self.performance_history) == 0:
            return {
                'error': '백테스트 데이터가 없습니다',
                'total_return': 0.0,
                'benchmark_return': 0.0,
                'excess_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'final_value': self.initial_capital,
                'num_trades': 0,
                'avg_crisis_level': 0.0,
                'immune_system_stats': {},
                'performance_history': [],
                'weights_history': [],
                'crisis_history': [],
                'decision_history': []
            }
        
        # 성과 계산
        total_return = (self.portfolio_value / self.initial_capital) - 1
        
        # 벤치마크 계산 (균등 가중)
        benchmark_prices = self.price_data.iloc[start_idx:end_idx]
        benchmark_returns = benchmark_prices.pct_change().mean(axis=1)
        benchmark_value = self.initial_capital * (1 + benchmark_returns).cumprod().iloc[-1]
        benchmark_return = (benchmark_value / self.initial_capital) - 1
        
        # 리스크 메트릭 계산
        portfolio_returns = pd.Series(self.performance_history).pct_change().dropna()
        
        if len(portfolio_returns) > 0:
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = (portfolio_returns.mean() * 252) / volatility if volatility > 0 else 0
            
            # 최대 낙폭
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        # 면역 시스템 통계
        immune_stats = self._get_immune_system_stats()
        
        return {
            'total_return': total_return,
            'benchmark_return': benchmark_return,
            'excess_return': total_return - benchmark_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': self.portfolio_value,
            'num_trades': len(self.decision_history),
            'avg_crisis_level': np.mean(self.crisis_history),
            'immune_system_stats': immune_stats,
            'performance_history': self.performance_history,
            'weights_history': self.weights_history,
            'crisis_history': self.crisis_history,
            'decision_history': self.decision_history
        }
    
    def _get_immune_system_stats(self) -> Dict[str, Any]:
        """면역 시스템 통계"""
        # B-Cell 성능
        bcell_stats = {}
        for specialty, bcell in self.bcells.items():
            bcell_stats[specialty] = bcell.get_performance_metrics()
        
        # 메모리 시스템 통계
        memory_stats = self.memory_system.get_system_statistics()
        
        # 리스크 관리 통계
        risk_stats = self.risk_manager.get_risk_report()
        
        return {
            'bcell_performance': bcell_stats,
            'memory_system': memory_stats,
            'risk_management': risk_stats,
            'tcell_activations': len(self.tcell.get_crisis_history())
        }
    
    def get_latest_explanation(self) -> Dict[str, Any]:
        """최신 의사결정 설명 반환"""
        if not self.decision_history:
            return {'error': '의사결정 히스토리가 없습니다'}
        
        latest_decision = self.decision_history[-1]
        
        return {
            'timestamp': latest_decision['timestamp'],
            'crisis_analysis': latest_decision['crisis_detection'],
            'memory_analysis': latest_decision['memory_recall'],
            'bcell_strategies': latest_decision['bcell_explanations'],
            'final_portfolio': {
                'weights': latest_decision['final_weights'],
                'assets': self.symbols
            },
            'system_reasoning': self._generate_system_reasoning(latest_decision)
        }
    
    def _generate_system_reasoning(self, decision: Dict[str, Any]) -> str:
        """시스템 의사결정 근거 생성"""
        reasoning_parts = []
        
        # 위기 분석
        crisis_level = decision['crisis_level']
        if crisis_level > 0.5:
            reasoning_parts.append(f"높은 위기 수준({crisis_level:.2f})으로 방어적 전략 활성화")
        elif crisis_level > 0.3:
            reasoning_parts.append(f"중간 위기 수준({crisis_level:.2f})으로 균형 전략 적용")
        else:
            reasoning_parts.append(f"낮은 위기 수준({crisis_level:.2f})으로 공격적 전략 허용")
        
        # B-Cell 활성화 분석
        active_bcells = []
        for specialty, explanation in decision['bcell_explanations'].items():
            if explanation['is_specialty_situation']:
                active_bcells.append(f"{specialty}({explanation['activation_strength']:.2f})")
        
        if active_bcells:
            reasoning_parts.append(f"활성화된 전문 전략: {', '.join(active_bcells)}")
        
        # 메모리 활용
        if decision['memory_recall']:
            similarity = decision['memory_recall']['similarity']
            reasoning_parts.append(f"과거 유사 경험 활용(유사도: {similarity:.2f})")
        
        return "; ".join(reasoning_parts)
    
    def reset(self):
        """시스템 리셋"""
        # 면역 시스템 리셋
        self.tcell.reset()
        for bcell in self.bcells.values():
            bcell.reset()
        self.memory_system.clear_all_memories()
        
        # 리스크 관리 리셋
        self.risk_manager.reset()
        
        # 상태 초기화
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = self.initial_capital
        self.step_count = 0
        
        # 히스토리 초기화
        self.performance_history.clear()
        self.weights_history.clear()
        self.crisis_history.clear()
        self.decision_history.clear()
        
        self.is_fitted = False
        
        print("시스템 리셋 완료")
    
    def export_results(self, filepath: str):
        """결과 내보내기"""
        if not self.decision_history:
            print("내보낼 결과가 없습니다")
            return
        
        # 결과 데이터 준비
        results_data = {
            'system_config': {
                'symbols': self.symbols,
                'initial_capital': self.initial_capital,
                'lookback_window': self.lookback_window,
                'rebalance_frequency': self.rebalance_frequency
            },
            'performance_summary': {
                'total_steps': self.step_count,
                'final_value': self.portfolio_value,
                'total_return': (self.portfolio_value / self.initial_capital) - 1
            },
            'immune_system_stats': self._get_immune_system_stats(),
            'decision_history': self.decision_history[-100:],  # 최근 100개만
            'export_timestamp': datetime.now().isoformat()
        }
        
        # JSON 저장
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"결과 저장 완료: {filepath}")