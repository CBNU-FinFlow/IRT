"""
시각화 도구
포트폴리오 성과 및 면역 시스템 상태 시각화
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 색상 팔레트 설정
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#F18F01',
    'warning': '#C73E1D',
    'info': '#6C757D',
    'tcell': '#FF6B6B',
    'bcell': '#4ECDC4',
    'memory': '#45B7D1',
    'crisis': '#E74C3C',
    'normal': '#27AE60'
}


class PortfolioPlotter:
    """포트폴리오 성과 시각화"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.setup_style()
    
    def setup_style(self):
        """스타일 설정"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_portfolio_performance(self, 
                                 portfolio_values: np.ndarray,
                                 benchmark_values: np.ndarray = None,
                                 dates: List[str] = None,
                                 title: str = "Portfolio Performance") -> plt.Figure:
        """
        포트폴리오 성과 시각화
        
        Args:
            portfolio_values: 포트폴리오 가치
            benchmark_values: 벤치마크 가치 (선택사항)
            dates: 날짜 목록
            title: 제목
            
        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 날짜 처리
        if dates is None:
            dates = range(len(portfolio_values))
        
        # 포트폴리오 성과 플롯
        ax.plot(dates, portfolio_values, 
               color=COLORS['primary'], linewidth=2, label='BIPD Portfolio')
        
        # 벤치마크 플롯
        if benchmark_values is not None:
            ax.plot(dates, benchmark_values, 
                   color=COLORS['secondary'], linewidth=2, 
                   linestyle='--', label='Benchmark')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 성과 통계 텍스트 박스
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        ax.text(0.02, 0.98, f'Total Return: {total_return:.2f}%', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
               verticalalignment='top')
        
        plt.tight_layout()
        return fig
    
    def plot_weight_evolution(self, 
                            weights_history: List[np.ndarray],
                            asset_names: List[str],
                            dates: List[str] = None,
                            title: str = "Portfolio Weight Evolution") -> plt.Figure:
        """
        포트폴리오 가중치 변화 시각화
        
        Args:
            weights_history: 가중치 히스토리
            asset_names: 자산 이름 목록
            dates: 날짜 목록
            title: 제목
            
        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 데이터 준비
        weights_df = pd.DataFrame(weights_history, columns=asset_names)
        
        if dates is None:
            dates = range(len(weights_history))
        
        # 스택 플롯
        ax.stackplot(dates, weights_df.T, 
                    labels=asset_names, alpha=0.7)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown(self, 
                     portfolio_values: np.ndarray,
                     dates: List[str] = None,
                     title: str = "Maximum Drawdown Analysis") -> plt.Figure:
        """
        최대 낙폭 시각화
        
        Args:
            portfolio_values: 포트폴리오 가치
            dates: 날짜 목록
            title: 제목
            
        Returns:
            matplotlib Figure 객체
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # 날짜 처리
        if dates is None:
            dates = range(len(portfolio_values))
        
        # 최대 낙폭 계산
        cumulative_returns = portfolio_values / portfolio_values[0]
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        
        # 포트폴리오 가치 플롯
        ax1.plot(dates, portfolio_values, color=COLORS['primary'], linewidth=2)
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 낙폭 플롯
        ax2.fill_between(dates, drawdown, 0, color=COLORS['warning'], alpha=0.7)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 최대 낙폭 표시
        max_dd = np.min(drawdown)
        ax2.text(0.02, 0.02, f'Max Drawdown: {max_dd:.2%}', 
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_risk_metrics(self, 
                         risk_metrics: Dict[str, float],
                         title: str = "Risk Metrics") -> plt.Figure:
        """
        리스크 메트릭 시각화
        
        Args:
            risk_metrics: 리스크 메트릭 딕셔너리
            title: 제목
            
        Returns:
            matplotlib Figure 객체
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # 변동성
        ax1.bar(['Volatility'], [risk_metrics.get('volatility', 0)], 
               color=COLORS['warning'])
        ax1.set_title('Annual Volatility', fontweight='bold')
        ax1.set_ylabel('Volatility (%)')
        
        # VaR
        var_95 = risk_metrics.get('var_95', 0)
        var_99 = risk_metrics.get('var_99', 0)
        ax2.bar(['VaR 95%', 'VaR 99%'], [var_95, var_99], 
               color=[COLORS['info'], COLORS['crisis']])
        ax2.set_title('Value at Risk', fontweight='bold')
        ax2.set_ylabel('VaR (%)')
        
        # 샤프 비율
        ax3.bar(['Sharpe Ratio'], [risk_metrics.get('sharpe_ratio', 0)], 
               color=COLORS['success'])
        ax3.set_title('Sharpe Ratio', fontweight='bold')
        ax3.set_ylabel('Ratio')
        
        # 집중도 리스크
        ax4.bar(['Concentration'], [risk_metrics.get('concentration_risk', 0)], 
               color=COLORS['secondary'])
        ax4.set_title('Concentration Risk (HHI)', fontweight='bold')
        ax4.set_ylabel('Concentration')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig


class ImmuneSystemPlotter:
    """면역 시스템 상태 시각화"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.setup_style()
    
    def setup_style(self):
        """스타일 설정"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_tcell_activation(self, 
                            activation_history: List[float],
                            crisis_levels: List[float],
                            dates: List[str] = None,
                            title: str = "T-Cell Activation Status") -> plt.Figure:
        """
        T-Cell 활성화 상태 시각화
        
        Args:
            activation_history: 활성화 히스토리
            crisis_levels: 위기 수준 히스토리
            dates: 날짜 목록
            title: 제목
            
        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 날짜 처리
        if dates is None:
            dates = range(len(activation_history))
        
        # 활성화 수준 플롯
        ax.plot(dates, activation_history, 
               color=COLORS['tcell'], linewidth=2, label='T-Cell Activation')
        
        # 위기 수준 플롯
        ax.plot(dates, crisis_levels, 
               color=COLORS['crisis'], linewidth=2, 
               linestyle='--', label='Crisis Level')
        
        # 임계값 선
        ax.axhline(y=0.15, color='red', linestyle=':', alpha=0.7, label='Activation Threshold')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Activation Level', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_bcell_specialization(self, 
                                bcell_activations: Dict[str, List[float]],
                                dates: List[str] = None,
                                title: str = "B-Cell Specialization Activation") -> plt.Figure:
        """
        B-Cell 전문화 활성화 시각화
        
        Args:
            bcell_activations: B-Cell 활성화 딕셔너리
            dates: 날짜 목록
            title: 제목
            
        Returns:
            matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 날짜 처리
        if dates is None:
            dates = range(len(next(iter(bcell_activations.values()))))
        
        # 각 B-Cell 활성화 플롯
        for specialty, activations in bcell_activations.items():
            ax.plot(dates, activations, linewidth=2, 
                   label=f'{specialty} B-Cell')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Activation Strength', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_memory_usage(self, 
                         memory_stats: Dict[str, Any],
                         title: str = "Memory System Usage Status") -> plt.Figure:
        """
        메모리 시스템 사용 현황 시각화
        
        Args:
            memory_stats: 메모리 통계
            title: 제목
            
        Returns:
            matplotlib Figure 객체
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # 메모리 사용률
        utilization = memory_stats.get('memory_utilization', 0)
        ax1.pie([utilization, 1-utilization], 
               labels=['Used', 'Available'], 
               colors=[COLORS['memory'], COLORS['info']],
               autopct='%1.1f%%', startangle=90)
        ax1.set_title('Memory Utilization', fontweight='bold')
        
        # 회상 성공률
        match_rate = memory_stats.get('match_rate', 0)
        ax2.bar(['Recall Success Rate'], [match_rate], 
               color=COLORS['success'])
        ax2.set_title('Recall Success Rate', fontweight='bold')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_ylim(0, 1)
        
        # 성공/실패 패턴
        success_count = memory_stats.get('success_patterns', 0)
        failure_count = memory_stats.get('failure_patterns', 0)
        ax3.bar(['Success Patterns', 'Failure Patterns'], [success_count, failure_count], 
               color=[COLORS['success'], COLORS['warning']])
        ax3.set_title('Stored Patterns', fontweight='bold')
        ax3.set_ylabel('Pattern Count')
        
        # 평균 성과
        avg_performance = memory_stats.get('avg_performance', 0)
        ax4.bar(['Average Performance'], [avg_performance], 
               color=COLORS['primary'])
        ax4.set_title('Average Performance', fontweight='bold')
        ax4.set_ylabel('Performance (%)')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_immune_system_overview(self, 
                                   system_stats: Dict[str, Any],
                                   title: str = "Immune System Overview") -> plt.Figure:
        """
        면역 시스템 종합 현황 시각화
        
        Args:
            system_stats: 시스템 통계
            title: 제목
            
        Returns:
            matplotlib Figure 객체
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # T-Cell 상태
        ax1 = fig.add_subplot(gs[0, 0])
        tcell_activation = system_stats.get('tcell_activation', 0)
        colors = [COLORS['crisis'] if tcell_activation > 0.5 else COLORS['normal']]
        ax1.bar(['T-Cell'], [tcell_activation], color=colors)
        ax1.set_title('T-Cell Activation', fontweight='bold')
        ax1.set_ylabel('Activation Level')
        ax1.set_ylim(0, 1)
        
        # B-Cell 활성화
        ax2 = fig.add_subplot(gs[0, 1])
        bcell_data = system_stats.get('bcell_activations', {})
        if bcell_data:
            specialties = list(bcell_data.keys())
            activations = list(bcell_data.values())
            ax2.bar(specialties, activations, color=COLORS['bcell'])
            ax2.set_title('B-Cell Specialization', fontweight='bold')
            ax2.set_ylabel('Activation Strength')
            ax2.tick_params(axis='x', rotation=45)
        
        # 메모리 사용률
        ax3 = fig.add_subplot(gs[0, 2])
        memory_utilization = system_stats.get('memory_utilization', 0)
        ax3.pie([memory_utilization, 1-memory_utilization], 
               colors=[COLORS['memory'], COLORS['info']],
               autopct='%1.1f%%', startangle=90)
        ax3.set_title('Memory Utilization', fontweight='bold')
        
        # 위기 수준 히스토리
        ax4 = fig.add_subplot(gs[1, :])
        crisis_history = system_stats.get('crisis_history', [])
        if crisis_history:
            dates = range(len(crisis_history))
            ax4.plot(dates, crisis_history, color=COLORS['crisis'], linewidth=2)
            ax4.fill_between(dates, crisis_history, alpha=0.3, color=COLORS['crisis'])
            ax4.set_title('Crisis Level History', fontweight='bold')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Crisis Level')
            ax4.grid(True, alpha=0.3)
        
        # 성과 분포
        ax5 = fig.add_subplot(gs[2, 0])
        performance_history = system_stats.get('performance_history', [])
        if performance_history:
            ax5.hist(performance_history, bins=20, color=COLORS['primary'], alpha=0.7)
            ax5.set_title('Performance Distribution', fontweight='bold')
            ax5.set_xlabel('Return')
            ax5.set_ylabel('Frequency')
        
        # 리스크 메트릭
        ax6 = fig.add_subplot(gs[2, 1])
        risk_metrics = system_stats.get('risk_metrics', {})
        if risk_metrics:
            metrics = ['Volatility', 'Sharpe', 'Max DD']
            values = [
                risk_metrics.get('volatility', 0),
                risk_metrics.get('sharpe_ratio', 0),
                abs(risk_metrics.get('max_drawdown', 0))
            ]
            ax6.bar(metrics, values, color=[COLORS['warning'], COLORS['success'], COLORS['crisis']])
            ax6.set_title('Risk Metrics', fontweight='bold')
            ax6.tick_params(axis='x', rotation=45)
        
        # 시스템 건강도
        ax7 = fig.add_subplot(gs[2, 2])
        health_score = system_stats.get('system_health', 0.5)
        colors = ['red' if health_score < 0.3 else 'yellow' if health_score < 0.7 else 'green']
        ax7.bar(['Health Score'], [health_score], color=colors)
        ax7.set_title('System Health', fontweight='bold')
        ax7.set_ylabel('Health Score')
        ax7.set_ylim(0, 1)
        
        plt.suptitle(title, fontsize=18, fontweight='bold')
        return fig


def save_plot(fig: plt.Figure, filepath: str, dpi: int = 300):
    """
    플롯 저장
    
    Args:
        fig: matplotlib Figure 객체
        filepath: 저장 경로
        dpi: 해상도
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)