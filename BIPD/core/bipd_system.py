"""
BIPD 통합 시스템
생체면역 기반 포트폴리오 방어 시스템의 핵심 구현
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings

from agents.tcell import TCell
from agents.bcell import BCell
from memory.memory_cell import MemorySystem
from risk.risk_management import RiskManager, SafeCalculations
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
        self.state_encoder = TemporalStateEncoder()
        
        # 안전 계산 유틸리티
        self.safe_calc = SafeCalculations()
    
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
        print("T-Cell 훈련 중...")
        self.tcell.fit(train_features.values)
        
        # 시간적 패턴 학습
        print("시간적 패턴 학습 중...")
        self.pattern_detector.fit(train_features.values)
        
        # 적응형 시간 윈도우 초기화
        self.time_window_manager.initialize(train_features.values)
        
        self.is_fitted = True
        print("시스템 훈련 완료")
        return self
    
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
        for i in range(start_idx, end_idx):
            if verbose and i % 50 == 0:
                progress = (i - start_idx) / (end_idx - start_idx) * 100
                print(f"진행률: {progress:.1f}% ({i}/{end_idx})")
            
            try:
                step_result = self.step(i)
                
                # 리밸런싱 주기 확인
                if self.step_count % self.rebalance_frequency == 0:
                    if verbose:
                        print(f"리밸런싱: {step_result['timestamp']}, "
                              f"위기수준: {step_result['crisis_level']:.3f}, "
                              f"포트폴리오 가치: {self.portfolio_value:.2f}")
                
            except Exception as e:
                print(f"Error at step {i}: {str(e)}")
                continue
        
        # 결과 생성
        results = self._generate_backtest_results(start_idx, end_idx)
        
        print(f"백테스트 완료. 총 수익률: {results['total_return']:.2%}")
        
        return results
    
    def _generate_backtest_results(self, start_idx: int, end_idx: int) -> Dict[str, Any]:
        """백테스트 결과 생성"""
        if len(self.performance_history) == 0:
            return {'error': '백테스트 데이터가 없습니다'}
        
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