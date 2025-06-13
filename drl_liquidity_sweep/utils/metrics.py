"""Performance metrics calculation utilities."""

from typing import Tuple

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe ratio.
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    if len(excess_returns) < 2:
        return 0.0
        
    sharpe = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns, ddof=1)
    return sharpe


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sortino ratio.
    
    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    if len(excess_returns) < 2:
        return 0.0
        
    downside_returns = np.where(returns < 0, returns, 0)
    downside_std = np.std(downside_returns, ddof=1)
    
    if downside_std == 0:
        return 0.0
        
    sortino = np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_std
    return sortino


def calculate_mar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Calculate MAR (Managed Assets Ratio) ratio.
    
    MAR = Annualized Return / Maximum Drawdown
    
    Args:
        returns: Array of period returns
        periods_per_year: Number of periods in a year
        
    Returns:
        MAR ratio
    """
    if len(returns) < 2:
        return 0.0
        
    cumulative = np.cumprod(1 + returns)
    drawdown = np.maximum.accumulate(cumulative) - cumulative
    max_drawdown = np.max(drawdown)
    
    if max_drawdown == 0:
        return 0.0
        
    annual_return = np.power(cumulative[-1], periods_per_year/len(returns)) - 1
    mar = annual_return / max_drawdown
    return mar


def calculate_trade_stats(
    trades: pd.DataFrame,
) -> dict:
    """Calculate trade statistics.
    
    Args:
        trades: DataFrame with columns: entry_price, exit_price, pnl
        
    Returns:
        Dict with trade statistics
    """
    stats = {
        "total_trades": len(trades),
        "win_rate": np.mean(trades["pnl"] > 0),
        "avg_win": trades[trades["pnl"] > 0]["pnl"].mean(),
        "avg_loss": trades[trades["pnl"] < 0]["pnl"].mean(),
        "profit_factor": abs(trades[trades["pnl"] > 0]["pnl"].sum() / 
                           trades[trades["pnl"] < 0]["pnl"].sum()),
    }
    return stats 