"""
면역 포트폴리오 백테스터
기존 main.py의 ImmunePortfolioBacktester 클래스를 모듈화
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import os
import json
import pickle
from tqdm import tqdm

from core.bipd_system import BIPDSystem
from utils.decision_analyzer import DecisionAnalyzer
from utils.data_loader import download_market_data, get_default_symbols
from visualization.html_dashboard import HTMLDashboardGenerator
from visualization.immune_visualization import ImmuneSystemVisualizer


class ImmunePortfolioBacktester:
    """
    면역 포트폴리오 백테스터
    """
    
    def __init__(self, 
                 symbols: List[str],
                 train_start: str,
                 train_end: str,
                 test_start: str,
                 test_end: str,
                 initial_capital: float = 100000,
                 results_dir: str = "results"):
        
        self.symbols = symbols
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.initial_capital = initial_capital
        self.results_dir = results_dir
        
        # 디렉토리 생성
        os.makedirs(results_dir, exist_ok=True)
        
        # 데이터 디렉토리 설정
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # 컴포넌트 초기화
        self.bipd_system = None
        self.decision_analyzer = None
        self.data = None
        self.train_data = None
        self.test_data = None
        self.train_features = None
        self.test_features = None
        
        # 결과 저장용
        self.backtest_results = []
        self.immune_system = None
        
        print(f"백테스터 초기화 완료: {len(symbols)}개 종목, {train_start}~{test_end}")
    
    def load_data(self):
        """
        데이터 로드 및 전처리
        """
        with tqdm(total=1, desc="Data Loading and Preprocessing", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            try:
                # 데이터 다운로드 (data/ 폴더에 저장)
                self.data = download_market_data(
                    symbols=self.symbols,
                    start_date=self.train_start,
                    end_date=self.test_end,
                    cache_dir="data"
                )
                
                # 데이터 분할
                self.train_data = self.data["prices"][self.train_start:self.train_end]
                self.test_data = self.data["prices"][self.test_start:self.test_end]
                self.train_features = self.data["features"][self.train_start:self.train_end]
                self.test_features = self.data["features"][self.test_start:self.test_end]
                
                # 데이터 정리
                self.train_data = self._clean_data(self.train_data)
                self.test_data = self._clean_data(self.test_data)
                self.train_features = self._clean_data(self.train_features)
                self.test_features = self._clean_data(self.test_features)
                
                pbar.set_postfix({
                    'Train': f'{self.train_data.shape[0]}d',
                    'Test': f'{self.test_data.shape[0]}d',
                    'Features': f'{self.train_features.shape[1]}f'
                })
                pbar.update(1)
                
            except Exception as e:
                pbar.set_postfix({'Error': str(e)[:30]})
                raise
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 정리"""
        # 무한값 및 결측값 처리
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(method='ffill').fillna(method='bfill')
        data = data.fillna(0)
        
        # 최소 길이 확인
        if len(data) < 100:
            raise ValueError(f"데이터 길이가 부족합니다: {len(data)}")
        
        return data
    
    def initialize_system(self, seed: int = 42):
        """
        BIPD 시스템 초기화
        
        Args:
            seed: 랜덤 시드
        """
        with tqdm(total=1, desc="BIPD System Initialization", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            
            # 시드 설정
            np.random.seed(seed)
            
            # BIPD 시스템 초기화
            self.bipd_system = BIPDSystem(
                symbols=self.symbols,
                initial_capital=self.initial_capital,
                lookback_window=252,
                rebalance_frequency=5
            )
            
            # 의사결정 분석기 초기화
            self.decision_analyzer = DecisionAnalyzer(
                output_dir=os.path.join(self.results_dir, "decision_analysis")
            )
            
            # 데이터 로딩 (시스템 내부)
            if self.data is None:
                self.load_data()
            
            # 시스템에 데이터 설정
            self.bipd_system.price_data = self.data["prices"]
            self.bipd_system.returns_data = self.data["prices"].pct_change().dropna()
            self.bipd_system.technical_indicators = self.data["features"]
            self.bipd_system.market_features = self.data["features"]
            
            pbar.update(1)
    
    def run_training(self):
        """
        시스템 훈련 실행
        """
        if self.bipd_system is None:
            raise ValueError("시스템이 초기화되지 않았습니다. initialize_system()을 먼저 호출하세요.")
        
        print("시스템 훈련 시작...")
        
        try:
            # 훈련 데이터 비율 계산
            train_end_ratio = len(self.train_data) / len(self.data["prices"])
            
            # BIPD 시스템 훈련
            self.bipd_system.fit(train_end_ratio=train_end_ratio)
            
            print("시스템 훈련 완료")
            
        except Exception as e:
            print(f"시스템 훈련 실패: {str(e)}")
            raise
    
    def backtest_single_run(self, 
                          seed: int = 42,
                          return_model: bool = False,
                          use_learning_bcells: bool = True,
                          logging_level: str = "full") -> Tuple[pd.Series, Optional[BIPDSystem]]:
        """
        단일 백테스트 실행
        
        Args:
            seed: 랜덤 시드
            return_model: 모델 반환 여부
            use_learning_bcells: 학습 가능한 B-Cell 사용 여부
            logging_level: 로깅 수준 ("full", "sample", "minimal")
            
        Returns:
            포트폴리오 수익률 시리즈, 선택적으로 BIPD 시스템
        """
        print(f"백테스트 시작 (시드: {seed}, 로깅: {logging_level})")
        
        # 시스템 초기화
        self.initialize_system(seed)
        
        # 훈련 실행
        self.run_training()
        
        # 백테스트 실행
        try:
            # 테스트 데이터 인덱스 범위 계산
            test_start_idx = len(self.train_data)
            test_end_idx = len(self.data["prices"])
            
            print(f"백테스트 기간: {test_start_idx} ~ {test_end_idx} ({test_end_idx - test_start_idx}일)")
            
            # 백테스트 실행
            results = self.bipd_system.run_backtest(
                start_idx=test_start_idx,
                end_idx=test_end_idx,
                verbose=(logging_level == "full")
            )
            
            # 의사결정 로깅
            if logging_level in ["full", "sample"]:
                self._log_decisions(results, logging_level)
            
            # 포트폴리오 수익률 계산
            portfolio_returns = self._calculate_portfolio_returns(results)
            
            # 결과 저장
            self.backtest_results.append({
                "seed": seed,
                "results": results,
                "portfolio_returns": portfolio_returns,
                "timestamp": datetime.now().isoformat()
            })
            
            if 'total_return' in results:
                print(f"백테스트 완료 - 총 수익률: {results['total_return']:.2%}")
            else:
                print("백테스트 완료 - 결과 처리 중 오류 발생")
            
            if return_model:
                return portfolio_returns, self.bipd_system
            else:
                return portfolio_returns, None
                
        except Exception as e:
            print(f"백테스트 실행 실패: {str(e)}")
            raise
    
    def _log_decisions(self, results: Dict[str, Any], logging_level: str):
        """
        의사결정 로깅
        
        Args:
            results: 백테스트 결과
            logging_level: 로깅 수준
        """
        if logging_level == "minimal":
            return
        
        decision_history = results.get("decision_history", [])
        
        # 샘플링 비율 설정
        if logging_level == "sample":
            # 10개 중 1개만 로깅
            decision_history = decision_history[::10]
        
        for decision in decision_history:
            try:
                # 의사결정 분석기에 로깅
                market_features = np.array(decision.get("market_features", []))
                if len(market_features) == 0:
                    # 빈 배열인 경우 기본값 생성
                    market_features = np.zeros(12)
                
                # T-Cell 분석 데이터 생성 (crisis_detection이 없는 경우)
                tcell_analysis = decision.get("crisis_detection", {
                    "crisis_level": decision.get("crisis_level", 0.0),
                    "decision_reasoning": "위기 감지 결과"
                })
                
                self.decision_analyzer.log_decision(
                    date=decision["timestamp"],
                    market_features=market_features,
                    tcell_analysis=tcell_analysis,
                    bcell_decisions=decision.get("bcell_explanations", {}),
                    final_weights=decision.get("final_weights", np.ones(10)/10),
                    portfolio_return=decision.get("portfolio_return", 0.0),
                    crisis_level=decision.get("crisis_level", 0.0)
                )
            except Exception as e:
                print(f"의사결정 로깅 중 오류: {str(e)}")
                continue
    
    def _calculate_portfolio_returns(self, results: Dict[str, Any]) -> pd.Series:
        """
        포트폴리오 수익률 계산
        
        Args:
            results: 백테스트 결과
            
        Returns:
            포트폴리오 수익률 시리즈
        """
        performance_history = results.get("performance_history", [])
        
        if not performance_history:
            return pd.Series([], dtype=float)
        
        # 수익률 계산
        portfolio_values = pd.Series(performance_history)
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        return portfolio_returns
    
    def calculate_metrics(self, portfolio_returns: pd.Series) -> Dict[str, float]:
        """
        성과 지표 계산
        
        Args:
            portfolio_returns: 포트폴리오 수익률 시리즈
            
        Returns:
            성과 지표 딕셔너리
        """
        if len(portfolio_returns) == 0:
            return {
                "Total Return": 0.0,
                "Sharpe Ratio": 0.0,
                "Max Drawdown": 0.0,
                "Volatility": 0.0,
                "Win Rate": 0.0
            }
        
        # 기본 통계
        total_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # 샤프 비율 (무위험 수익률 2% 가정)
        excess_returns = portfolio_returns - 0.02/252
        sharpe_ratio = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
        
        # 최대 낙폭
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 승률
        win_rate = (portfolio_returns > 0).mean()
        
        return {
            "Total Return": total_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Volatility": volatility,
            "Win Rate": win_rate
        }
    
    def run_multiple_backtests(self,
                             n_runs: int = 5,
                             save_results: bool = True,
                             use_learning_bcells: bool = True,
                             logging_level: str = "sample",
                             base_seed: int = 42) -> Dict[str, Any]:
        """
        다중 백테스트 실행
        
        Args:
            n_runs: 실행 횟수
            save_results: 결과 저장 여부
            use_learning_bcells: 학습 가능한 B-Cell 사용 여부
            logging_level: 로깅 수준
            base_seed: 기본 시드
            
        Returns:
            다중 백테스트 결과
        """
        print(f"\n다중 백테스트 시작: {n_runs}회 실행")
        
        all_results = []
        all_metrics = []
        
        for i in tqdm(range(n_runs), desc="백테스트 진행"):
            seed = base_seed + i
            
            try:
                portfolio_returns, model = self.backtest_single_run(
                    seed=seed,
                    return_model=(i == 0),  # 첫 번째 모델만 반환
                    use_learning_bcells=use_learning_bcells,
                    logging_level=logging_level
                )
                
                # 성과 지표 계산
                metrics = self.calculate_metrics(portfolio_returns)
                metrics["seed"] = seed
                metrics["run_id"] = i + 1
                
                all_results.append(portfolio_returns)
                all_metrics.append(metrics)
                
                # 첫 번째 모델 저장
                if i == 0 and model is not None:
                    self.immune_system = model
                
            except Exception as e:
                print(f"백테스트 {i+1} 실행 실패: {str(e)}")
                continue
        
        # 결과 통계 계산
        summary_stats = self._calculate_summary_statistics(all_metrics)
        
        # 결과 저장
        if save_results:
            self._save_multiple_results(all_results, all_metrics, summary_stats)
        
        return {
            "individual_results": all_results,
            "individual_metrics": all_metrics,
            "summary_statistics": summary_stats,
            "n_successful_runs": len(all_results)
        }
    
    def _calculate_summary_statistics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        요약 통계 계산
        
        Args:
            all_metrics: 개별 백테스트 지표들
            
        Returns:
            요약 통계
        """
        if not all_metrics:
            return {}
        
        metrics_df = pd.DataFrame(all_metrics)
        
        summary = {}
        for metric in ["Total Return", "Sharpe Ratio", "Max Drawdown", "Volatility", "Win Rate"]:
            if metric in metrics_df.columns:
                summary[f"{metric}_mean"] = metrics_df[metric].mean()
                summary[f"{metric}_std"] = metrics_df[metric].std()
                summary[f"{metric}_min"] = metrics_df[metric].min()
                summary[f"{metric}_max"] = metrics_df[metric].max()
        
        return summary
    
    def _save_multiple_results(self, 
                             all_results: List[pd.Series],
                             all_metrics: List[Dict[str, float]],
                             summary_stats: Dict[str, float]):
        """
        다중 백테스트 결과 저장
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.results_dir, f"multiple_backtest_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # 개별 결과 저장
        for i, (returns, metrics) in enumerate(zip(all_results, all_metrics)):
            returns.to_csv(os.path.join(results_dir, f"returns_run_{i+1}.csv"))
        
        # 지표 저장
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(os.path.join(results_dir, "metrics_summary.csv"), index=False)
        
        # 요약 통계 저장
        with open(os.path.join(results_dir, "summary_statistics.json"), 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"다중 백테스트 결과 저장 완료: {results_dir}")
    
    def save_comprehensive_analysis(self, 
                                  start_date: str = None,
                                  end_date: str = None) -> Tuple[str, str]:
        """
        종합 분석 저장
        
        Args:
            start_date: 분석 시작 날짜
            end_date: 분석 종료 날짜
            
        Returns:
            (JSON 파일 경로, 마크다운 파일 경로)
        """
        if self.decision_analyzer is None:
            raise ValueError("의사결정 분석기가 초기화되지 않았습니다.")
        
        # 분석 보고서 생성
        json_path = self.decision_analyzer.save_analysis_report(
            start_date=start_date,
            end_date=end_date
        )
        
        # 마크다운 보고서 생성
        md_path = self._generate_markdown_report(json_path, start_date, end_date)
        
        return json_path, md_path
    
    def _generate_markdown_report(self, json_path: str, start_date: str, end_date: str) -> str:
        """
        마크다운 보고서 생성
        """
        # JSON 데이터 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        # 마크다운 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = json_path.replace('.json', '.md')
        
        summary = report_data.get("summary", {})
        
        md_content = f"""# BIPD 시스템 종합 분석 보고서

## 분석 개요
- **생성 시간**: {report_data['metadata']['generated_at']}
- **분석 기간**: {start_date} ~ {end_date}
- **총 의사결정 수**: {report_data['metadata']['total_decisions']}
- **위기 감지 수**: {report_data['metadata']['crisis_detections']}

## 성과 요약
- **평균 수익률**: {summary.get('performance_statistics', {}).get('avg_return', 0):.4f}
- **변동성**: {summary.get('performance_statistics', {}).get('volatility', 0):.4f}
- **샤프 비율**: {summary.get('performance_statistics', {}).get('sharpe_ratio', 0):.4f}
- **양의 수익률 비율**: {summary.get('performance_statistics', {}).get('positive_returns', 0):.2%}

## 위기 관리
- **평균 위기 수준**: {summary.get('crisis_statistics', {}).get('avg_crisis_level', 0):.3f}
- **최대 위기 수준**: {summary.get('crisis_statistics', {}).get('max_crisis_level', 0):.3f}
- **위기 에피소드 수**: {summary.get('crisis_statistics', {}).get('crisis_episodes', 0)}

## T-Cell 활성화
- **평균 활성화 수준**: {summary.get('tcell_statistics', {}).get('avg_activation', 0):.3f}
- **최대 활성화 수준**: {summary.get('tcell_statistics', {}).get('max_activation', 0):.3f}
- **활성화 빈도**: {summary.get('tcell_statistics', {}).get('activation_frequency', 0):.2%}

## B-Cell 전문화
"""
        
        # B-Cell 통계 추가
        bcell_stats = summary.get('bcell_statistics', {})
        for cell_type, stats in bcell_stats.items():
            md_content += f"""
### {cell_type.capitalize()} B-Cell
- **평균 활성화**: {stats.get('avg_activation', 0):.3f}
- **최대 활성화**: {stats.get('max_activation', 0):.3f}
- **활성화 빈도**: {stats.get('activation_frequency', 0):.2%}
"""
        
        # 파일 저장
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return md_path
    
    def save_analysis_results(self, 
                            start_date: str = None,
                            end_date: str = None) -> Tuple[str, str, str]:
        """
        분석 결과 저장 (HTML 대시보드 + 면역 시스템 시각화)
        
        Args:
            start_date: 분석 시작 날짜
            end_date: 분석 종료 날짜
            
        Returns:
            (JSON 파일 경로, 마크다운 파일 경로, HTML 대시보드 경로)
        """
        # 기본 분석 보고서 생성
        json_path, md_path = self.save_comprehensive_analysis(start_date, end_date)
        
        # HTML 대시보드 생성
        html_path = self._generate_html_dashboard(json_path)
        
        # 면역 시스템 시각화 생성
        if self.immune_system is not None:
            self._generate_immune_visualizations()
        
        return json_path, md_path, html_path
    
    def _generate_html_dashboard(self, json_path: str) -> str:
        """
        HTML 대시보드 생성
        """
        # JSON 데이터 로드
        with open(json_path, 'r', encoding='utf-8') as f:
            analysis_report = json.load(f)
        
        # HTML 대시보드 생성기 초기화
        dashboard_generator = HTMLDashboardGenerator()
        
        # 대시보드 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = os.path.join(self.results_dir, f"dashboard_{timestamp}.html")
        
        dashboard_generator.generate_dashboard(
            analysis_report=analysis_report,
            output_path=html_path
        )
        
        return html_path
    
    def _generate_immune_visualizations(self):
        """
        면역 시스템 시각화 생성
        """
        try:
            # 면역 시스템 시각화 생성기 초기화
            visualizer = ImmuneSystemVisualizer()
            
            # 시각화 데이터 준비
            if hasattr(self.immune_system, 'decision_history') and self.immune_system.decision_history:
                decision_data = self.immune_system.decision_history[-100:]  # 최근 100개
                
                # 시각화 생성
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                viz_dir = os.path.join(self.results_dir, f"immune_visualizations_{timestamp}")
                os.makedirs(viz_dir, exist_ok=True)
                
                # 시각화 파일 생성 (create_visualizations 함수 호출)
                visualizer.create_visualizations(
                    decision_data=decision_data,
                    output_dir=viz_dir
                )
                
                print(f"면역 시스템 시각화 생성 완료: {viz_dir}")
            
        except Exception as e:
            print(f"면역 시스템 시각화 생성 중 오류: {str(e)}")
    
    def reset(self):
        """
        백테스터 리셋
        """
        self.bipd_system = None
        self.decision_analyzer = None
        self.backtest_results.clear()
        self.immune_system = None
        
        print("백테스터 리셋 완료")
    
    def get_latest_results(self) -> Optional[Dict[str, Any]]:
        """
        최신 백테스트 결과 반환
        """
        if not self.backtest_results:
            return None
        
        return self.backtest_results[-1]