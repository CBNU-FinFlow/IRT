"""
BIPD (Behavioral Immune Portfolio Defense) 시스템 실행 스크립트
생체면역시스템 기반 포트폴리오 관리 시스템

Usage:
    python main.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import argparse
from tqdm import tqdm
import time

# 모듈 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.bipd_system import BIPDSystem
from core.backtester import ImmunePortfolioBacktester
from utils.data_loader import get_default_symbols
from utils.decision_analyzer import DecisionAnalyzer
from visualization.plotter import PortfolioPlotter, ImmuneSystemPlotter, save_plot
# from visualization.html_dashboard import HTMLDashboardGenerator  # 제거됨
from visualization.immune_visualization import ImmuneSystemVisualizer

warnings.filterwarnings("ignore")


def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description="BIPD 시스템 실행")
    
    parser.add_argument("--symbols", nargs="+", default=None,
                      help="투자 대상 심볼 목록 (기본: 미국 대형주 10개)")
    parser.add_argument("--start-date", type=str, default="2008-01-01",
                      help="백테스트 시작 날짜 (기본: 2008-01-01)")
    parser.add_argument("--end-date", type=str, default="2024-12-31",
                      help="백테스트 종료 날짜 (기본: 2024-12-31)")
    parser.add_argument("--initial-capital", type=float, default=100000,
                      help="초기 자본 (기본: 100,000)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                      help="훈련 데이터 비율 (기본: 0.8)")
    parser.add_argument("--rebalance-freq", type=int, default=5,
                      help="리밸런싱 주기 (기본: 5일)")
    parser.add_argument("--save-results", action="store_true",
                      help="결과 저장 여부")
    parser.add_argument("--results-dir", type=str, default="results",
                      help="결과 저장 디렉토리 (기본: results)")
    parser.add_argument("--verbose", action="store_true",
                      help="상세 출력 여부")
    
    return parser.parse_args()


def create_results_directory(results_dir: str) -> str:
    """결과 디렉토리 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(results_dir, f"analysis_{timestamp}")
    os.makedirs(full_path, exist_ok=True)
    return full_path


def print_system_info():
    """시스템 정보 출력"""
    print("\n" + "=" * 60)
    print("BIPD: Behavioral Immune Portfolio Defense System")
    print("=" * 60)
    print("생체면역시스템 기반 적응형 포트폴리오 관리 시스템")
    print("├─ T-Cell: 위기 감지 및 설명가능성")
    print("├─ B-Cell: 전문화된 투자 전략 생성")
    print("├─ Memory Cell: 과거 경험 학습 및 재활용")
    print("└─ Risk Management: 포트폴리오 제약 및 리스크 관리")
    print("=" * 60 + "\n")


def print_comprehensive_results(metrics: dict, symbols: list):
    """종합 백테스트 결과 출력"""
    print("\n" + "=" * 45)
    print("BIPD 포트폴리오 성과 요약")
    print("=" * 45)
    
    # 성과 지표 시각화
    print(f"총 수익률    : {metrics['Total Return']:>8.2%}")
    print(f"샤프 비율    : {metrics['Sharpe Ratio']:>8.3f}")
    print(f"최대 낙폭    : {metrics['Max Drawdown']:>8.2%}")
    print(f"연간 변동성  : {metrics['Volatility']:>8.2%}")
    print(f"승률        : {metrics['Win Rate']:>8.2%}")


def print_backtest_results(results: dict, symbols: list):
    """백테스트 결과 출력"""
    print("\n" + "=" * 50)
    print("백테스트 결과")
    print("=" * 50)
    
    print(f"총 수익률: {results['total_return']:.2%}")
    print(f"벤치마크 수익률: {results['benchmark_return']:.2%}")
    print(f"초과 수익률: {results['excess_return']:.2%}")
    print(f"연간 변동성: {results['volatility']:.2%}")
    print(f"샤프 비율: {results['sharpe_ratio']:.3f}")
    print(f"최대 낙폭: {results['max_drawdown']:.2%}")
    print(f"최종 포트폴리오 가치: ${results['final_value']:,.2f}")
    print(f"총 거래 수: {results['num_trades']}")
    print(f"평균 위기 수준: {results['avg_crisis_level']:.3f}")
    
    # 면역 시스템 통계
    immune_stats = results['immune_system_stats']
    print(f"\n면역 시스템 통계:")
    print(f"- T-Cell 활성화 횟수: {immune_stats['tcell_activations']}")
    print(f"- 메모리 시스템 총 기억: {immune_stats['memory_system']['total_memories']}")
    print(f"- 메모리 회상 성공률: {immune_stats['memory_system']['overall_match_rate']:.2%}")
    
    # B-Cell 성능
    print(f"\nB-Cell 전문화 성능:")
    for specialty, metrics in immune_stats['bcell_performance'].items():
        print(f"- {specialty}: 평균 보상 {metrics['avg_reward']:.4f}, "
              f"업데이트 {metrics['update_count']}회")
    
    # 리스크 관리 통계
    risk_stats = immune_stats['risk_management']
    print(f"\n리스크 관리 통계:")
    print(f"- 포지션 제한 위반: {risk_stats['risk_metrics']['position_violations']}")
    print(f"- 터노버 제한 위반: {risk_stats['risk_metrics']['turnover_violations']}")
    print(f"- 집중도 제한 위반: {risk_stats['risk_metrics']['concentration_violations']}")
    
    # 최종 포트폴리오 구성
    if results['weights_history']:
        final_weights = results['weights_history'][-1]
        print(f"\n최종 포트폴리오 구성:")
        for i, (symbol, weight) in enumerate(zip(symbols, final_weights)):
            print(f"- {symbol}: {weight:.2%}")


def create_visualizations(backtester: ImmunePortfolioBacktester, symbols: list, save_dir: str):
    """시각화 생성"""
    print("\n시각화 생성 중...")
    
    try:
        # 최신 결과 가져오기
        latest_results = backtester.get_latest_results()
        if not latest_results:
            print("시각화할 결과가 없습니다.")
            return
        
        portfolio_returns = latest_results['portfolio_returns']
        
        # 포트폴리오 플로터
        portfolio_plotter = PortfolioPlotter()
        
        # 성과 차트
        portfolio_values = (1 + portfolio_returns).cumprod() * backtester.initial_capital
        
        fig1 = portfolio_plotter.plot_portfolio_performance(
            portfolio_values.values,
            title="BIPD 포트폴리오 성과"
        )
        save_plot(fig1, os.path.join(save_dir, "portfolio_performance.png"))
        
        # 낙폭 차트
        fig2 = portfolio_plotter.plot_drawdown(
            portfolio_values.values,
            title="포트폴리오 낙폭 분석"
        )
        save_plot(fig2, os.path.join(save_dir, "drawdown_analysis.png"))
        
        # 리스크 메트릭
        metrics = backtester.calculate_metrics(portfolio_returns)
        fig3 = portfolio_plotter.plot_risk_metrics(
            metrics,
            title="리스크 메트릭 분석"
        )
        save_plot(fig3, os.path.join(save_dir, "risk_metrics.png"))
        
        print(f"Visualization completed: {save_dir}")
        
    except Exception as e:
        print(f"시각화 생성 중 오류: {str(e)}")


def create_visualizations(results: dict, symbols: list, save_dir: str):
    """시각화 생성 및 저장"""
    print("\nCreating visualizations...")
    
    # 포트폴리오 플로터
    portfolio_plotter = PortfolioPlotter()
    
    # 성과 차트
    portfolio_values = np.array(results['performance_history'])
    
    # 벤치마크 계산 (간단한 균등가중 가정)
    benchmark_values = np.linspace(100000, 100000 * (1 + results['benchmark_return']), 
                                  len(portfolio_values))
    
    fig1 = portfolio_plotter.plot_portfolio_performance(
        portfolio_values, benchmark_values,
        title="BIPD Portfolio Performance vs Benchmark"
    )
    save_plot(fig1, os.path.join(save_dir, "portfolio_performance.png"))
    
    # 가중치 변화
    if results['weights_history']:
        weights_sample = results['weights_history'][::10]  # 10개씩 샘플링
        fig2 = portfolio_plotter.plot_weight_evolution(
            weights_sample, symbols,
            title="Portfolio Weight Evolution"
        )
        save_plot(fig2, os.path.join(save_dir, "weight_evolution.png"))
    
    # 낙폭 차트
    fig3 = portfolio_plotter.plot_drawdown(
        portfolio_values,
        title="Portfolio Drawdown Analysis"
    )
    save_plot(fig3, os.path.join(save_dir, "drawdown_analysis.png"))
    
    # 면역 시스템 플로터
    immune_plotter = ImmuneSystemPlotter()
    
    # 위기 감지 차트
    crisis_history = results['crisis_history']
    activation_history = [min(c * 2, 1.0) for c in crisis_history]  # 활성화 추정
    
    fig4 = immune_plotter.plot_tcell_activation(
        activation_history, crisis_history,
        title="T-Cell Crisis Detection and Activation"
    )
    save_plot(fig4, os.path.join(save_dir, "tcell_activation.png"))
    
    # B-Cell 전문화 분석
    bcell_data = {}
    for specialty, metrics in results['immune_system_stats']['bcell_performance'].items():
        # 간단한 활성화 히스토리 추정
        avg_activation = max(0, metrics['avg_reward'] * 10)
        bcell_data[specialty] = [avg_activation] * min(100, len(crisis_history))
    
    if bcell_data:
        fig5 = immune_plotter.plot_bcell_specialization(
            bcell_data,
            title="B-Cell Specialization Activation Pattern"
        )
        save_plot(fig5, os.path.join(save_dir, "bcell_specialization.png"))
    
    # 메모리 시스템 사용 현황
    memory_stats = results['immune_system_stats']['memory_system']
    fig6 = immune_plotter.plot_memory_usage(
        memory_stats,
        title="Memory System Usage Status"
    )
    save_plot(fig6, os.path.join(save_dir, "memory_usage.png"))
    
    print(f"Visualization completed: {save_dir}")


def save_detailed_results(results: dict, bipd_system: BIPDSystem, save_dir: str):
    """상세 결과 저장"""
    print("\n상세 결과 저장 중...")
    
    # 시스템 결과 JSON 저장
    results_file = os.path.join(save_dir, "backtest_results.json")
    bipd_system.export_results(results_file)
    
    # 성과 데이터 CSV 저장
    performance_df = pd.DataFrame({
        'portfolio_value': results['performance_history'],
        'crisis_level': results['crisis_history']
    })
    performance_df.to_csv(os.path.join(save_dir, "performance_data.csv"), index=False)
    
    # 가중치 히스토리 저장
    if results['weights_history']:
        weights_df = pd.DataFrame(results['weights_history'], columns=bipd_system.symbols)
        weights_df.to_csv(os.path.join(save_dir, "weights_history.csv"), index=False)
    
    # 최신 설명 저장
    latest_explanation = bipd_system.get_latest_explanation()
    if 'error' not in latest_explanation:
        import json
        with open(os.path.join(save_dir, "latest_explanation.json"), 'w', encoding='utf-8') as f:
            json.dump(latest_explanation, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"상세 결과 저장 완료: {save_dir}")


def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    print_system_info()
    
    # 기본 설정
    symbols = args.symbols or get_default_symbols()
    print(f"\n투자 대상: {symbols}")
    print(f"기간: {args.start_date} ~ {args.end_date}")
    print(f"초기 자본: ${args.initial_capital:,.2f}")
    print(f"훈련 데이터 비율: {args.train_ratio:.1%}")
    print(f"리밸런싱 주기: {args.rebalance_freq}일")
    
    # 훈련/테스트 기간 계산
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    total_days = (end_date - start_date).days
    train_days = int(total_days * args.train_ratio)
    
    train_start = start_date.strftime("%Y-%m-%d")
    train_end = (start_date + pd.Timedelta(days=train_days)).strftime("%Y-%m-%d")
    test_start = (start_date + pd.Timedelta(days=train_days + 1)).strftime("%Y-%m-%d")
    test_end = end_date.strftime("%Y-%m-%d")
    
    print(f"\n훈련 기간: {train_start} ~ {train_end}")
    print(f"테스트 기간: {test_start} ~ {test_end}")
    
    try:
        # ImmunePortfolioBacktester 초기화
        print("\n백테스터 초기화 중...")
        backtester = ImmunePortfolioBacktester(
            symbols=symbols,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            initial_capital=args.initial_capital,
            results_dir=args.results_dir
        )
        
        # 데이터 로드
        print("\n데이터 로딩 중...")
        backtester.load_data()
        
        # 단일 백테스트 실행
        print("\n백테스트 실행 중...")
        portfolio_returns, immune_system = backtester.backtest_single_run(
            seed=42,
            return_model=True,
            use_learning_bcells=True,
            logging_level="full" if args.verbose else "sample"
        )
        
        # 성과 계산
        metrics = backtester.calculate_metrics(portfolio_returns)
        print_comprehensive_results(metrics, symbols)
        
        # 결과 저장 (기본적으로 항상 저장)
        if True:  # args.save_results:
            # main_original 스타일로 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = f"results/analysis_{timestamp}"
            os.makedirs(results_dir, exist_ok=True)
            
            # 기본 백테스트 결과 저장
            print("\n결과 저장 중...")
            
            # 포트폴리오 성과 저장
            metrics = backtester.calculate_metrics(portfolio_returns)
            results_summary = {
                "backtest_period": f"{test_start} ~ {test_end}",
                "total_return": metrics["Total Return"],
                "sharpe_ratio": metrics["Sharpe Ratio"], 
                "max_drawdown": metrics["Max Drawdown"],
                "volatility": metrics["Volatility"],
                "win_rate": metrics["Win Rate"],
                "symbols": symbols,
                "initial_capital": args.initial_capital
            }
            
            # JSON 저장
            import json
            with open(f"{results_dir}/backtest_results.json", 'w') as f:
                json.dump(results_summary, f, indent=2, default=str)
            
            # CSV 저장 (포트폴리오 수익률)
            portfolio_returns.to_csv(f"{results_dir}/portfolio_returns.csv")
            
            print(f"\n결과 저장 완료: {results_dir}")
            print(f"- 백테스트 결과: {results_dir}/backtest_results.json")
            print(f"- 포트폴리오 수익률: {results_dir}/portfolio_returns.csv")
        
        # 최신 의사결정 설명 출력
        print("\n" + "=" * 30)
        print("최신 의사결정 설명")
        print("=" * 30)
        
        if immune_system:
            explanation = immune_system.get_latest_explanation()
            if 'error' not in explanation:
                print(f"시점        : {explanation['timestamp']}")
                print(f"위기 분석   : {explanation['crisis_analysis']['decision_reasoning']}")
                print(f"시스템 판단 : {explanation['system_reasoning']}")
                
                # 활성화된 전문 전략 출력
                active_strategies = []
                for specialty, exp in explanation['bcell_strategies'].items():
                    if exp['is_specialty_situation']:
                        active_strategies.append(f"{specialty} (강도: {exp['activation_strength']:.2f})")
                
                if active_strategies:
                    print(f"활성화된 전문 전략: {', '.join(active_strategies)}")
                
                # 메모리 활용
                if explanation.get('memory_analysis'):
                    memory_info = explanation['memory_analysis']
                    if memory_info and 'similarity' in memory_info:
                        print(f"메모리 활용: 유사도 {memory_info['similarity']:.3f}의 과거 경험 활용")
                    else:
                        print("메모리 활용: 없음")
        
        # 다중 실행 안정성 검증 (선택사항)
        if args.verbose:
            print("\n" + "=" * 30)
            print("다중 실행 안정성 검증")
            print("=" * 30)
            
            multiple_results = backtester.run_multiple_backtests(
                n_runs=3,
                save_results=True,
                use_learning_bcells=True,
                logging_level="sample",
                base_seed=42
            )
            
            # 안정성 통계 출력
            summary_stats = multiple_results['summary_statistics']
            print(f"평균 총 수익률: {summary_stats.get('Total Return_mean', 0):.2%} ± {summary_stats.get('Total Return_std', 0):.2%}")
            print(f"평균 샤프 비율: {summary_stats.get('Sharpe Ratio_mean', 0):.3f} ± {summary_stats.get('Sharpe Ratio_std', 0):.3f}")
            print(f"평균 최대 낙폭: {summary_stats.get('Max Drawdown_mean', 0):.2%} ± {summary_stats.get('Max Drawdown_std', 0):.2%}")
        
        print("\n" + "=" * 50)
        print("BIPD 시스템 실행 완료")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
