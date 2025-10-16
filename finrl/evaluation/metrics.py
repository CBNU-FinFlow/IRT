# src/analysis/metrics.py

"""
성과 지표: 포트폴리오 성과 측정 도구

목적: 포트폴리오 성과의 정량적 평가
의존: scipy.stats
사용처: PortfolioEnv, monitor.py, backtest.py
역할: 일관된 성과 측정 제공

구현 내용:
- Sharpe Ratio (위험 조정 수익률)
- Maximum Drawdown (최대 낙폭)
- Calmar Ratio (낙폭 대비 수익률)
- Sortino Ratio (하방 위험 조정)
- CVaR/VaR (리스크 측정)
- 회전율 및 거래 비용 분석
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List, Dict
from scipy import stats

def calculate_sharpe_ratio(returns: Union[np.ndarray, pd.Series],
                         risk_free_rate: float = 0.02,
                         periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio with numerical stability

    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        sharpe_ratio: Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    # 극단값 처리
    if len(returns) == 1:
        # 단일 데이터 포인트의 경우 Sharpe ratio 계산 불가
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year

    # Epsilon for numerical stability
    eps = 1e-8
    std_returns = np.std(excess_returns)

    # 표준편차가 극도로 작을 때 처리
    if std_returns < eps:
        # 변동성이 거의 없는 경우
        mean_excess = np.mean(excess_returns)
        if abs(mean_excess) < eps:
            return 0.0  # 수익률도 0에 가까운 경우
        else:
            # 수익률은 있지만 변동성이 없는 경우 (이론적 최댓값으로 제한)
            return np.sign(mean_excess) * 10.0  # 최대 Sharpe ratio를 10으로 제한

    sharpe = np.mean(excess_returns) / std_returns
    annualized_sharpe = sharpe * np.sqrt(periods_per_year)

    # 극단값 클리핑 (현실적인 범위: -5 ~ 5)
    return float(np.clip(annualized_sharpe, -5.0, 5.0))

def calculate_sortino_ratio(returns: Union[np.ndarray, pd.Series],
                          target_return: float = 0.0,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (downside risk-adjusted return) with numerical stability

    Args:
        returns: Return series
        target_return: Target return threshold
        periods_per_year: Number of periods per year

    Returns:
        sortino_ratio: Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    if len(returns) == 1:
        # 단일 데이터 포인트의 경우 Sortino ratio 계산 불가
        return 0.0

    excess_returns = returns - target_return / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        # 손실이 없는 경우 (매우 좋은 성과)
        return 10.0  # 최대값으로 제한

    # Epsilon for numerical stability
    eps = 1e-8
    downside_std = np.std(downside_returns)

    if downside_std < eps:
        # Downside 변동성이 거의 없는 경우
        mean_excess = np.mean(excess_returns)
        if abs(mean_excess) < eps:
            return 0.0  # 수익률도 0에 가까운 경우
        else:
            # 수익률은 있지만 downside 변동성이 없는 경우
            return np.sign(mean_excess) * 10.0  # 최대값으로 제한

    sortino = np.mean(excess_returns) / downside_std
    annualized_sortino = sortino * np.sqrt(periods_per_year)

    # 극단값 클리핑 (현실적인 범위: -5 ~ 10, Sortino는 일반적으로 Sharpe보다 높음)
    return float(np.clip(annualized_sortino, -5.0, 10.0))

def calculate_calmar_ratio(returns: Union[np.ndarray, pd.Series],
                         periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (return / max drawdown)
    
    Args:
        returns: Return series
        periods_per_year: Number of periods per year
        
    Returns:
        calmar_ratio: Calmar ratio
    """
    if len(returns) == 0:
        return 0.0

    cumulative_returns = (1 + returns).cumprod()

    # pandas Series를 numpy array로 변환 (일관성 확보)
    if isinstance(cumulative_returns, pd.Series):
        cumulative_returns = cumulative_returns.values

    annual_return = cumulative_returns[-1] ** (periods_per_year / len(returns)) - 1
    max_dd = calculate_max_drawdown(cumulative_returns)
    
    if max_dd == 0:
        return float('inf')
    
    return float(annual_return / abs(max_dd))

def calculate_max_drawdown(values: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate maximum drawdown
    
    Args:
        values: Value series (prices or cumulative returns)
        
    Returns:
        max_drawdown: Maximum drawdown (negative value)
    """
    if len(values) == 0:
        return 0.0
    
    if isinstance(values, pd.Series):
        values = values.values
    
    cumulative = values / values[0]
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    
    return float(np.min(drawdown))

def calculate_max_drawdown_duration(values: Union[np.ndarray, pd.Series]) -> int:
    """
    Calculate maximum drawdown duration in periods
    
    Args:
        values: Value series
        
    Returns:
        max_duration: Maximum drawdown duration
    """
    if len(values) == 0:
        return 0
    
    if isinstance(values, pd.Series):
        values = values.values
    
    cumulative = values / values[0]
    running_max = np.maximum.accumulate(cumulative)
    
    drawdown_periods = []
    in_drawdown = False
    current_duration = 0
    
    for i in range(len(cumulative)):
        if cumulative[i] < running_max[i]:
            if not in_drawdown:
                in_drawdown = True
                current_duration = 1
            else:
                current_duration += 1
        else:
            if in_drawdown:
                drawdown_periods.append(current_duration)
                in_drawdown = False
                current_duration = 0
    
    if in_drawdown:
        drawdown_periods.append(current_duration)
    
    return max(drawdown_periods) if drawdown_periods else 0

def calculate_cvar(returns: Union[np.ndarray, pd.Series],
                  alpha: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
    
    Args:
        returns: Return series
        alpha: Confidence level (e.g., 0.05 for 95% CVaR)
        
    Returns:
        cvar: CVaR at given confidence level
    """
    if len(returns) == 0:
        return 0.0
    
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    var = np.percentile(returns, alpha * 100)
    cvar = np.mean(returns[returns <= var])
    
    return float(cvar)

def calculate_var(returns: Union[np.ndarray, pd.Series],
                 alpha: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR)
    
    Args:
        returns: Return series
        alpha: Confidence level
        
    Returns:
        var: VaR at given confidence level
    """
    if len(returns) == 0:
        return 0.0
    
    return float(np.percentile(returns, alpha * 100))

def calculate_turnover(weights_history: np.ndarray) -> float:
    """
    Calculate average portfolio turnover
    
    Args:
        weights_history: Time series of portfolio weights (T x N)
        
    Returns:
        avg_turnover: Average turnover per period
    """
    if len(weights_history) < 2:
        return 0.0
    
    turnovers = []
    for i in range(1, len(weights_history)):
        turnover = np.sum(np.abs(weights_history[i] - weights_history[i-1])) / 2
        turnovers.append(turnover)
    
    return float(np.mean(turnovers))

def calculate_concentration_index(weights: np.ndarray) -> float:
    """
    Calculate portfolio concentration (Herfindahl index)
    
    Args:
        weights: Portfolio weights
        
    Returns:
        concentration: Herfindahl concentration index
    """
    return float(np.sum(weights ** 2))

def calculate_effective_assets(weights: np.ndarray,
                              threshold: float = 0.01) -> int:
    """
    Calculate number of effective assets (weights above threshold)
    
    Args:
        weights: Portfolio weights
        threshold: Minimum weight threshold
        
    Returns:
        n_effective: Number of effective assets
    """
    return int(np.sum(weights > threshold))

def calculate_information_ratio(returns: Union[np.ndarray, pd.Series],
                              benchmark_returns: Union[np.ndarray, pd.Series],
                              periods_per_year: int = 252) -> float:
    """
    Calculate Information Ratio
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year
        
    Returns:
        ir: Information ratio
    """
    if len(returns) != len(benchmark_returns):
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
    
    excess_returns = returns - benchmark_returns
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    ir = np.mean(excess_returns) / np.std(excess_returns)
    return float(ir * np.sqrt(periods_per_year))

def calculate_tracking_error(returns: Union[np.ndarray, pd.Series],
                           benchmark_returns: Union[np.ndarray, pd.Series],
                           periods_per_year: int = 252) -> float:
    """
    Calculate tracking error (standard deviation of excess returns)
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year
        
    Returns:
        tracking_error: Annualized tracking error
    """
    if len(returns) != len(benchmark_returns):
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
    
    excess_returns = returns - benchmark_returns
    return float(np.std(excess_returns) * np.sqrt(periods_per_year))

def calculate_alpha_beta(returns: Union[np.ndarray, pd.Series],
                       benchmark_returns: Union[np.ndarray, pd.Series],
                       risk_free_rate: float = 0.02,
                       periods_per_year: int = 252) -> Tuple[float, float]:
    """
    Calculate alpha and beta using linear regression
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Market returns
        risk_free_rate: Risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        alpha: Jensen's alpha (annualized)
        beta: Portfolio beta
    """
    if len(returns) != len(benchmark_returns):
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
    
    if len(returns) < 2:
        return 0.0, 1.0
    
    # Excess returns
    excess_returns = returns - risk_free_rate / periods_per_year
    excess_benchmark = benchmark_returns - risk_free_rate / periods_per_year
    
    # Linear regression
    beta, alpha, _, _, _ = stats.linregress(excess_benchmark, excess_returns)
    
    # Annualize alpha
    alpha = alpha * periods_per_year
    
    return float(alpha), float(beta)

def calculate_omega_ratio(returns: Union[np.ndarray, pd.Series],
                        threshold: float = 0.0) -> float:
    """
    Calculate Omega ratio
    
    Args:
        returns: Return series
        threshold: Threshold return
        
    Returns:
        omega: Omega ratio
    """
    if len(returns) == 0:
        return 1.0
    
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    
    if losses == 0:
        return float('inf')
    
    return float(gains / losses)

def calculate_tail_ratio(returns: Union[np.ndarray, pd.Series],
                       percentile: float = 95) -> float:
    """
    Calculate tail ratio (ratio of right tail to left tail)
    
    Args:
        returns: Return series
        percentile: Percentile for tail definition
        
    Returns:
        tail_ratio: Tail ratio
    """
    if len(returns) == 0:
        return 1.0
    
    right_tail = np.percentile(returns, percentile)
    left_tail = np.percentile(returns, 100 - percentile)
    
    if abs(left_tail) < 1e-10:
        return float('inf')
    
    return float(abs(right_tail / left_tail))

def calculate_win_rate(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate win rate (percentage of positive returns)
    
    Args:
        returns: Return series
        
    Returns:
        win_rate: Win rate (0 to 1)
    """
    if len(returns) == 0:
        return 0.0
    
    return float(np.mean(returns > 0))

def calculate_profit_factor(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate profit factor (gross profit / gross loss)
    
    Args:
        returns: Return series
        
    Returns:
        profit_factor: Profit factor
    """
    if len(returns) == 0:
        return 1.0
    
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    
    if losses == 0:
        return float('inf')
    
    return float(gains / losses)

def calculate_recovery_factor(returns: Union[np.ndarray, pd.Series],
                            initial_value: float = 1.0) -> float:
    """
    Calculate recovery factor (net profit / max drawdown)
    
    Args:
        returns: Return series
        initial_value: Initial portfolio value
        
    Returns:
        recovery_factor: Recovery factor
    """
    if len(returns) == 0:
        return 0.0
    
    cumulative = (1 + returns).cumprod()
    final_value = cumulative.iloc[-1] if isinstance(cumulative, pd.Series) else cumulative[-1]
    net_profit = (final_value - 1) * initial_value
    max_dd = calculate_max_drawdown(cumulative * initial_value)
    
    if max_dd == 0:
        return float('inf')
    
    return float(net_profit / abs(max_dd))

def calculate_ulcer_index(returns: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Ulcer Index (measures downside volatility)
    
    Args:
        returns: Return series
        
    Returns:
        ulcer_index: Ulcer index
    """
    if len(returns) == 0:
        return 0.0
    
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown_pct = ((cumulative - running_max) / running_max) * 100
    
    ulcer = np.sqrt(np.mean(drawdown_pct ** 2))
    return float(ulcer)


class MetricsCalculator:
    """
    중앙화된 메트릭 계산 클래스
    
    모든 메트릭 계산 함수를 클래스 메서드로 제공
    """
    
    @staticmethod
    def calculate_sharpe_ratio(returns: Union[np.ndarray, pd.Series],
                             risk_free_rate: float = 0.02,
                             periods_per_year: int = 252) -> float:
        """Sharpe ratio 계산"""
        return calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    
    @staticmethod
    def calculate_sortino_ratio(returns: Union[np.ndarray, pd.Series],
                              target_return: float = 0.0,
                              periods_per_year: int = 252) -> float:
        """Sortino ratio 계산"""
        return calculate_sortino_ratio(returns, target_return, periods_per_year)
    
    @staticmethod
    def calculate_calmar_ratio(returns: Union[np.ndarray, pd.Series],
                             periods_per_year: int = 252) -> float:
        """Calmar ratio 계산"""
        return calculate_calmar_ratio(returns, periods_per_year)
    
    @staticmethod
    def calculate_max_drawdown(values: Union[np.ndarray, pd.Series]) -> float:
        """최대 낙폭 계산"""
        return calculate_max_drawdown(values)
    
    @staticmethod
    def calculate_max_drawdown_duration(values: Union[np.ndarray, pd.Series]) -> int:
        """최대 낙폭 기간 계산"""
        return calculate_max_drawdown_duration(values)
    
    @staticmethod
    def calculate_cvar(returns: Union[np.ndarray, pd.Series],
                      alpha: float = 0.05) -> float:
        """CVaR 계산"""
        return calculate_cvar(returns, alpha)
    
    @staticmethod
    def calculate_var(returns: Union[np.ndarray, pd.Series],
                     alpha: float = 0.05) -> float:
        """VaR 계산"""
        return calculate_var(returns, alpha)
    
    @staticmethod
    def calculate_turnover(weights_history: np.ndarray) -> float:
        """포트폴리오 회전율 계산"""
        return calculate_turnover(weights_history)
    
    @staticmethod
    def calculate_concentration_index(weights: np.ndarray) -> float:
        """포트폴리오 집중도 계산"""
        return calculate_concentration_index(weights)
    
    @staticmethod
    def calculate_effective_assets(weights: np.ndarray,
                                  threshold: float = 0.01) -> int:
        """유효 자산 수 계산"""
        return calculate_effective_assets(weights, threshold)
    
    @staticmethod
    def calculate_information_ratio(returns: Union[np.ndarray, pd.Series],
                                  benchmark_returns: Union[np.ndarray, pd.Series],
                                  periods_per_year: int = 252) -> float:
        """Information ratio 계산"""
        return calculate_information_ratio(returns, benchmark_returns, periods_per_year)
    
    @staticmethod
    def calculate_tracking_error(returns: Union[np.ndarray, pd.Series],
                               benchmark_returns: Union[np.ndarray, pd.Series],
                               periods_per_year: int = 252) -> float:
        """추적 오차 계산"""
        return calculate_tracking_error(returns, benchmark_returns, periods_per_year)
    
    @staticmethod
    def calculate_alpha_beta(returns: Union[np.ndarray, pd.Series],
                           benchmark_returns: Union[np.ndarray, pd.Series],
                           risk_free_rate: float = 0.02,
                           periods_per_year: int = 252) -> Tuple[float, float]:
        """알파와 베타 계산"""
        return calculate_alpha_beta(returns, benchmark_returns, risk_free_rate, periods_per_year)
    
    @staticmethod
    def calculate_omega_ratio(returns: Union[np.ndarray, pd.Series],
                            threshold: float = 0.0) -> float:
        """Omega ratio 계산"""
        return calculate_omega_ratio(returns, threshold)
    
    @staticmethod
    def calculate_tail_ratio(returns: Union[np.ndarray, pd.Series],
                           percentile: float = 95) -> float:
        """Tail ratio 계산"""
        return calculate_tail_ratio(returns, percentile)
    
    @staticmethod
    def calculate_win_rate(returns: Union[np.ndarray, pd.Series]) -> float:
        """승률 계산"""
        return calculate_win_rate(returns)
    
    @staticmethod
    def calculate_profit_factor(returns: Union[np.ndarray, pd.Series]) -> float:
        """Profit factor 계산"""
        return calculate_profit_factor(returns)
    
    @staticmethod
    def calculate_recovery_factor(returns: Union[np.ndarray, pd.Series],
                                initial_value: float = 1.0) -> float:
        """Recovery factor 계산"""
        return calculate_recovery_factor(returns, initial_value)
    
    @staticmethod
    def calculate_ulcer_index(returns: Union[np.ndarray, pd.Series]) -> float:
        """Ulcer index 계산"""
        return calculate_ulcer_index(returns)
    
    def calculate_all_metrics(self, 
                            returns: Union[np.ndarray, pd.Series],
                            values: Optional[Union[np.ndarray, pd.Series]] = None,
                            weights_history: Optional[np.ndarray] = None,
                            benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None,
                            risk_free_rate: float = 0.02,
                            periods_per_year: int = 252) -> Dict[str, float]:
        """
        모든 메트릭을 한번에 계산
        
        Args:
            returns: 수익률 시계열
            values: 가치 시계열 (옵션)
            weights_history: 포트폴리오 가중치 히스토리 (옵션)
            benchmark_returns: 벤치마크 수익률 (옵션)
            risk_free_rate: 무위험 이자율
            periods_per_year: 연간 기간 수
            
        Returns:
            메트릭 딕셔너리
        """
        metrics = {}
        
        # 기본 메트릭
        metrics['sharpe_ratio'] = self.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        metrics['sortino_ratio'] = self.calculate_sortino_ratio(returns, 0.0, periods_per_year)
        metrics['calmar_ratio'] = self.calculate_calmar_ratio(returns, periods_per_year)
        
        # 리스크 메트릭
        if values is not None:
            metrics['max_drawdown'] = self.calculate_max_drawdown(values)
            metrics['max_drawdown_duration'] = self.calculate_max_drawdown_duration(values)
        else:
            cumulative = (1 + returns).cumprod()
            metrics['max_drawdown'] = self.calculate_max_drawdown(cumulative)
            metrics['max_drawdown_duration'] = self.calculate_max_drawdown_duration(cumulative)
        
        metrics['var_95'] = self.calculate_var(returns, 0.05)
        metrics['cvar_95'] = self.calculate_cvar(returns, 0.05)
        
        # 포트폴리오 메트릭
        if weights_history is not None:
            metrics['turnover'] = self.calculate_turnover(weights_history)
            if len(weights_history) > 0:
                latest_weights = weights_history[-1]
                metrics['concentration'] = self.calculate_concentration_index(latest_weights)
                metrics['effective_assets'] = self.calculate_effective_assets(latest_weights)
        
        # 벤치마크 대비 메트릭
        if benchmark_returns is not None:
            metrics['information_ratio'] = self.calculate_information_ratio(
                returns, benchmark_returns, periods_per_year
            )
            metrics['tracking_error'] = self.calculate_tracking_error(
                returns, benchmark_returns, periods_per_year
            )
            alpha, beta = self.calculate_alpha_beta(
                returns, benchmark_returns, risk_free_rate, periods_per_year
            )
            metrics['alpha'] = alpha
            metrics['beta'] = beta
        
        # 추가 메트릭
        metrics['omega_ratio'] = self.calculate_omega_ratio(returns)
        metrics['tail_ratio'] = self.calculate_tail_ratio(returns)
        metrics['win_rate'] = self.calculate_win_rate(returns)
        metrics['profit_factor'] = self.calculate_profit_factor(returns)
        metrics['recovery_factor'] = self.calculate_recovery_factor(returns)
        metrics['ulcer_index'] = self.calculate_ulcer_index(returns)
        
        # 기본 통계
        metrics['total_return'] = float((1 + returns).prod() - 1)
        metrics['annual_return'] = float(
            (1 + metrics['total_return']) ** (periods_per_year / len(returns)) - 1
        )
        metrics['volatility'] = float(np.std(returns) * np.sqrt(periods_per_year))
        metrics['skewness'] = float(stats.skew(returns))
        metrics['kurtosis'] = float(stats.kurtosis(returns))
        
        return metrics