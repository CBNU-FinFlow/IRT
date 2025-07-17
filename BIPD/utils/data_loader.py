"""
데이터 로더
yfinance를 활용한 시장 데이터 다운로드 및 처리
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


def download_market_data(
    symbols: List[str],
    start_date: str = "2008-01-01",
    end_date: str = "2024-12-31",
    cache_dir: str = "data",
) -> Dict[str, pd.DataFrame]:
    """
    시장 데이터 다운로드 및 처리

    Args:
        symbols: 주식 심볼 목록
        start_date: 시작 날짜
        end_date: 종료 날짜
        cache_dir: 캐시 디렉토리

    Returns:
        {"prices": 가격 데이터, "features": 기술적 지표, "raw_data": 원시 데이터}
    """
    # 캐시 파일 경로
    cache_filename = f"market_data_{'_'.join(symbols)}_{start_date}_{end_date}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)

    # 캐시 확인
    if os.path.exists(cache_path):
        print(f"캐시된 데이터 로드: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 데이터 다운로드
    print(f"시장 데이터 다운로드 중: {symbols}")
    try:
        raw_data = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            progress=True,
            auto_adjust=False,  # 조정 가격과 원시 가격 모두 필요
            prepost=True,
            threads=True,
        )

        if raw_data.empty:
            raise ValueError("다운로드된 데이터가 없습니다.")

        # 데이터 처리
        processed_data = process_market_data(raw_data, symbols)

        # 캐시 저장
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(processed_data, f)

        print(f"데이터 처리 완료: {list(processed_data.keys())}")
        return processed_data

    except Exception as e:
        print(f"데이터 다운로드 실패: {str(e)}")
        raise


def process_market_data(
    raw_data: pd.DataFrame, symbols: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    시장 데이터 처리

    Args:
        raw_data: yfinance 원시 데이터
        symbols: 주식 심볼 목록

    Returns:
        {"prices": 가격 데이터, "features": 기술적 지표, "raw_data": 원시 데이터}
    """
    print("데이터 처리 중...")

    # 1. 기본 가격 데이터 추출
    prices = extract_price_data(raw_data, symbols)

    # 2. 기술적 지표 계산
    features = calculate_technical_indicators(raw_data, symbols)

    # 3. 데이터 정리
    prices = clean_data(prices)
    features = clean_data(features)

    return {"prices": prices, "features": features, "raw_data": raw_data}


def extract_price_data(raw_data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """
    가격 데이터 추출 (Adj Close 우선, 없으면 Close 사용)
    """
    if len(symbols) == 1:
        symbol = symbols[0]
        if "Adj Close" in raw_data.columns:
            prices = raw_data["Adj Close"].to_frame(symbol)
        elif "Close" in raw_data.columns:
            prices = raw_data["Close"].to_frame(symbol)
        else:
            raise ValueError("가격 데이터를 찾을 수 없습니다.")
    else:
        try:
            prices = raw_data["Adj Close"]
        except KeyError:
            try:
                prices = raw_data["Close"]
                print("주의: 'Adj Close' 없음, 'Close' 사용")
            except KeyError:
                # MultiIndex 구조에서 수동으로 추출
                price_data = {}
                for symbol in symbols:
                    if ("Adj Close", symbol) in raw_data.columns:
                        price_data[symbol] = raw_data[("Adj Close", symbol)]
                    elif ("Close", symbol) in raw_data.columns:
                        price_data[symbol] = raw_data[("Close", symbol)]
                    else:
                        print(f"경고: {symbol} 가격 데이터를 찾을 수 없습니다.")
                        continue

                if not price_data:
                    raise ValueError("사용 가능한 가격 데이터가 없습니다.")
                prices = pd.DataFrame(price_data)

    return prices


def calculate_technical_indicators(
    raw_data: pd.DataFrame, symbols: List[str]
) -> pd.DataFrame:
    """
    기술적 지표 계산
    """
    print("기술적 지표 계산 중...")

    features = {}

    for symbol in symbols:
        try:
            # OHLCV 데이터 추출
            ohlcv = extract_ohlcv_data(raw_data, symbol, symbols)

            # 개별 심볼 기술적 지표 계산
            symbol_features = calculate_symbol_technical_indicators(symbol, ohlcv)

            features[symbol] = symbol_features

        except Exception as e:
            print(f"[경고] {symbol} 기술적 지표 계산 중 오류 발생: {e}")
            continue

    # 전체 특성 데이터프레임 생성
    if not features:
        raise ValueError("기술적 지표를 계산할 수 있는 데이터가 없습니다.")

    all_features = pd.concat(features.values(), axis=1)

    # 시장 전체 지표 추가
    all_features = add_market_indicators(all_features, symbols)

    return all_features


def extract_ohlcv_data(
    raw_data: pd.DataFrame, symbol: str, symbols: List[str]
) -> Dict[str, pd.Series]:
    """
    개별 심볼의 OHLCV 데이터 추출
    """
    if len(symbols) == 1:
        return {
            "high": raw_data.get("High", raw_data.get("Close", pd.Series())),
            "low": raw_data.get("Low", raw_data.get("Close", pd.Series())),
            "close": raw_data.get("Adj Close", raw_data.get("Close", pd.Series())),
            "volume": raw_data.get("Volume", pd.Series(1, index=raw_data.index)),
        }
    else:
        return {
            "high": raw_data.get("High", {}).get(
                symbol, raw_data.get("Close", {}).get(symbol, pd.Series())
            ),
            "low": raw_data.get("Low", {}).get(
                symbol, raw_data.get("Close", {}).get(symbol, pd.Series())
            ),
            "close": raw_data.get("Adj Close", {}).get(
                symbol, raw_data.get("Close", {}).get(symbol, pd.Series())
            ),
            "volume": raw_data.get("Volume", {}).get(
                symbol, pd.Series(1, index=raw_data.index)
            ),
        }


def calculate_symbol_technical_indicators(
    symbol: str, ohlcv: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    개별 심볼의 기술적 지표 계산
    """
    high, low, close, volume = (
        ohlcv["high"],
        ohlcv["low"],
        ohlcv["close"],
        ohlcv["volume"],
    )

    symbol_features = pd.DataFrame(index=close.index)

    # 1. 가격 기반 지표
    symbol_features[f"{symbol}_returns"] = close.pct_change()
    symbol_features[f"{symbol}_volatility"] = (
        symbol_features[f"{symbol}_returns"].rolling(20).std()
    )
    symbol_features[f"{symbol}_sma_20"] = close.rolling(20).mean()
    symbol_features[f"{symbol}_sma_50"] = close.rolling(50).mean()
    symbol_features[f"{symbol}_price_sma20_ratio"] = (
        close / symbol_features[f"{symbol}_sma_20"]
    )
    symbol_features[f"{symbol}_price_sma50_ratio"] = (
        close / symbol_features[f"{symbol}_sma_50"]
    )

    # 2. 모멘텀 지표
    symbol_features[f"{symbol}_rsi"] = calculate_rsi(close, 14)
    symbol_features[f"{symbol}_momentum"] = close / close.shift(10) - 1

    # 3. 볼린저 밴드
    bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2)
    symbol_features[f"{symbol}_bb_position"] = (close - bb_lower) / (
        bb_upper - bb_lower
    )

    # 4. 거래량 지표
    symbol_features[f"{symbol}_volume_sma"] = volume.rolling(20).mean()
    symbol_features[f"{symbol}_volume_ratio"] = (
        volume / symbol_features[f"{symbol}_volume_sma"]
    )

    # 5. 변동성 지표
    symbol_features[f"{symbol}_high_low_ratio"] = (high - low) / close
    symbol_features[f"{symbol}_price_range"] = (high - low) / close.rolling(20).mean()

    # 6. MACD
    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()
    symbol_features[f"{symbol}_macd"] = ema_12 - ema_26
    symbol_features[f"{symbol}_macd_signal"] = (
        symbol_features[f"{symbol}_macd"].ewm(span=9).mean()
    )
    symbol_features[f"{symbol}_macd_histogram"] = (
        symbol_features[f"{symbol}_macd"] - symbol_features[f"{symbol}_macd_signal"]
    )

    return symbol_features


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(
    prices: pd.Series, period: int = 20, std_dev: float = 2
) -> Tuple[pd.Series, pd.Series]:
    """볼린저 밴드 계산"""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band


def add_market_indicators(features: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """
    시장 전체 지표 추가
    """
    print("시장 전체 지표 계산 중...")

    try:
        # 시장 전체 수익률 (동일 가중)
        return_cols = [col for col in features.columns if "_returns" in col]
        if return_cols:
            features["market_return"] = features[return_cols].mean(axis=1)
            features["market_volatility"] = features[return_cols].std(axis=1)

            # 상관계수 계산
            corr_values = []
            for i in range(len(features)):
                try:
                    window_data = features[return_cols].iloc[max(0, i - 19) : i + 1]
                    if len(window_data) >= 2:
                        corr_matrix = window_data.corr()
                        upper_tri = corr_matrix.where(
                            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                        )
                        corr_values.append(upper_tri.stack().mean())
                    else:
                        corr_values.append(0.0)
                except:
                    corr_values.append(0.0)

            features["market_correlation"] = pd.Series(
                corr_values, index=features.index
            )

        # VIX 대용 지표 (변동성의 변동성)
        vol_cols = [col for col in features.columns if "_volatility" in col]
        if vol_cols:
            features["vix_proxy"] = features[vol_cols].mean(axis=1).rolling(10).std()

        # 시장 스트레스 지수
        rsi_cols = [col for col in features.columns if "_rsi" in col]
        if rsi_cols:
            features["market_stress"] = features[rsi_cols].apply(
                lambda x: (x < 30).sum() + (x > 70).sum(), axis=1
            )

        # 모멘텀 지표
        momentum_cols = [col for col in features.columns if "_momentum" in col]
        if momentum_cols:
            features["market_momentum"] = features[momentum_cols].mean(axis=1)

        # 볼린저 밴드 위치
        bb_cols = [col for col in features.columns if "_bb_position" in col]
        if bb_cols:
            features["market_bb_position"] = features[bb_cols].mean(axis=1)

    except Exception as e:
        print(f"[경고] 시장 지표 계산 중 오류 발생: {e}")

    return features


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    데이터 정리 함수
    """
    # 무한값 제거
    data = data.replace([np.inf, -np.inf], np.nan)

    # 결측값 처리
    data = data.fillna(method="ffill").fillna(method="bfill")

    # 여전히 NaN인 값은 0으로 처리
    data = data.fillna(0)

    return data


def process_raw_data(raw_data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """
    원시 데이터 처리

    Args:
        raw_data: yfinance 원시 데이터
        symbols: 주식 심볼 목록

    Returns:
        처리된 데이터 DataFrame
    """
    processed_data = pd.DataFrame()

    try:
        # 단일 심볼 처리
        if len(symbols) == 1:
            symbol = symbols[0]
            if "Close" in raw_data.columns:
                processed_data[symbol] = raw_data["Close"]
            else:
                processed_data[symbol] = raw_data

        # 다중 심볼 처리
        else:
            if "Close" in raw_data.columns.get_level_values(0):
                # MultiIndex 컬럼 구조
                for symbol in symbols:
                    if ("Close", symbol) in raw_data.columns:
                        processed_data[symbol] = raw_data["Close", symbol]
                    elif symbol in raw_data.columns:
                        processed_data[symbol] = raw_data[symbol]
            else:
                # 단순 컬럼 구조
                for symbol in symbols:
                    if symbol in raw_data.columns:
                        processed_data[symbol] = raw_data[symbol]

        # 결측값 처리
        processed_data = processed_data.fillna(method="ffill").fillna(method="bfill")

        # 무한값 제거
        processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
        processed_data = processed_data.fillna(method="ffill").fillna(method="bfill")

        # 최소 데이터 길이 확인
        if len(processed_data) < 100:
            raise ValueError(f"데이터 길이가 부족합니다: {len(processed_data)}")

        # 인덱스 정리
        processed_data.index = pd.to_datetime(processed_data.index)
        processed_data = processed_data.sort_index()

        return processed_data

    except Exception as e:
        print(f"데이터 처리 실패: {str(e)}")
        raise


def calculate_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """
    수익률 계산

    Args:
        prices: 가격 데이터
        method: 계산 방법 ('simple' 또는 'log')

    Returns:
        수익률 DataFrame
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()

    return returns.dropna()


def calculate_technical_indicators(
    prices: pd.DataFrame, window: int = 20
) -> pd.DataFrame:
    """
    기술적 지표 계산

    Args:
        prices: 가격 데이터
        window: 계산 윈도우

    Returns:
        기술적 지표 DataFrame
    """
    indicators = pd.DataFrame(index=prices.index)

    for symbol in prices.columns:
        price_series = prices[symbol]

        # 이동평균
        indicators[f"{symbol}_SMA"] = price_series.rolling(window).mean()

        # RSI
        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators[f"{symbol}_RSI"] = 100 - (100 / (1 + rs))

        # 볼린저 밴드
        sma = price_series.rolling(window).mean()
        std = price_series.rolling(window).std()
        indicators[f"{symbol}_BB_Upper"] = sma + (std * 2)
        indicators[f"{symbol}_BB_Lower"] = sma - (std * 2)
        indicators[f"{symbol}_BB_Position"] = (
            price_series - indicators[f"{symbol}_BB_Lower"]
        ) / (indicators[f"{symbol}_BB_Upper"] - indicators[f"{symbol}_BB_Lower"])

        # 변동성
        returns = price_series.pct_change()
        indicators[f"{symbol}_Volatility"] = returns.rolling(window).std()

    return indicators.fillna(method="ffill").fillna(0)


def extract_market_features(
    prices: pd.DataFrame, indicators: pd.DataFrame, lookback: int = 20
) -> pd.DataFrame:
    """
    시장 특성 추출

    Args:
        prices: 가격 데이터
        indicators: 기술적 지표
        lookback: 참조 기간

    Returns:
        시장 특성 DataFrame
    """
    features = pd.DataFrame(index=prices.index)

    # 기본 통계
    returns = calculate_returns(prices)
    features["daily_return"] = returns.mean(axis=1)
    features["volatility"] = returns.std(axis=1)
    features["max_return"] = returns.max(axis=1)
    features["min_return"] = returns.min(axis=1)

    # 상관관계
    rolling_corr = returns.rolling(window=lookback).corr()
    features["avg_correlation"] = rolling_corr.groupby(level=0).mean().mean(axis=1)

    # 모멘텀
    price_momentum = prices.pct_change(periods=5)
    features["momentum"] = price_momentum.mean(axis=1)

    # 유동성 (거래량 기반 추정)
    features["liquidity"] = 1.0 - np.minimum(features["volatility"] / 0.05, 1.0)

    # 시장 스트레스
    features["market_stress"] = np.minimum(features["volatility"] / 0.03, 1.0)

    # 기술적 지표 평균
    rsi_columns = [col for col in indicators.columns if "RSI" in col]
    if rsi_columns:
        features["avg_rsi"] = indicators[rsi_columns].mean(axis=1)

    bb_columns = [col for col in indicators.columns if "BB_Position" in col]
    if bb_columns:
        features["avg_bb_position"] = indicators[bb_columns].mean(axis=1)

    vol_columns = [col for col in indicators.columns if "Volatility" in col]
    if vol_columns:
        features["avg_volatility"] = indicators[vol_columns].mean(axis=1)

    # 추세
    sma_columns = [col for col in indicators.columns if "SMA" in col]
    if sma_columns:
        current_prices = prices.values
        sma_values = indicators[sma_columns].values
        features["trend"] = np.nanmean(current_prices / sma_values, axis=1)

    # 결측값 처리
    features = features.fillna(method="ffill").fillna(0)

    return features


def split_data(
    data: pd.DataFrame, train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    데이터 분할

    Args:
        data: 전체 데이터
        train_ratio: 훈련 데이터 비율

    Returns:
        훈련 데이터, 테스트 데이터
    """
    split_point = int(len(data) * train_ratio)
    train_data = data.iloc[:split_point]
    test_data = data.iloc[split_point:]

    return train_data, test_data


def get_default_symbols() -> List[str]:
    """기본 주식 심볼 목록 반환"""
    return [
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "AMD",
        "META",
        "NVDA",
        "NFLX",
        "ADBE",
        "CRM",
    ]


def validate_data(data: pd.DataFrame, min_length: int = 100) -> bool:
    """
    데이터 검증

    Args:
        data: 검증할 데이터
        min_length: 최소 길이

    Returns:
        검증 결과
    """
    if data.empty:
        return False

    if len(data) < min_length:
        return False

    if data.isnull().sum().sum() > len(data) * 0.1:  # 10% 이상 결측값
        return False

    return True
