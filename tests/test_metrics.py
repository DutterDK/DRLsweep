"""Tests for performance metrics calculation."""

import numpy as np
import pandas as pd
import pytest

from drl_liquidity_sweep.utils.metrics import (calculate_mar_ratio,
                                             calculate_sharpe_ratio,
                                             calculate_sortino_ratio,
                                             calculate_trade_stats)


def test_calculate_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    # Test with positive returns
    returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    assert sharpe > 0
    
    # Test with negative returns
    returns = np.array([-0.01, -0.02, -0.01, -0.03, -0.01])
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    assert sharpe < 0
    
    # Test with risk-free rate
    returns = np.array([0.01, 0.02, 0.01, 0.02, 0.01])
    sharpe1 = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    sharpe2 = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    assert sharpe1 > sharpe2


def test_calculate_sortino_ratio():
    """Test Sortino ratio calculation."""
    # Test with mixed returns
    returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino != 0
    
    # Test with only positive returns
    returns = np.array([0.01, 0.02, 0.01, 0.03, 0.02])
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino > 0
    
    # Test with only negative returns
    returns = np.array([-0.01, -0.02, -0.01, -0.03, -0.02])
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino < 0


def test_calculate_mar_ratio():
    """Test MAR ratio calculation."""
    # Test with drawdown
    returns = np.array([0.01, -0.05, 0.02, 0.03, -0.02])
    mar = calculate_mar_ratio(returns)
    assert isinstance(mar, float)
    
    # Test with no drawdown
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    mar = calculate_mar_ratio(returns)
    assert mar == 0.0  # No drawdown means undefined MAR


def test_calculate_trade_stats():
    """Test trade statistics calculation."""
    trades = pd.DataFrame({
        "entry_price": [1.0, 1.1, 1.2, 1.3, 1.4],
        "exit_price": [1.1, 1.0, 1.3, 1.2, 1.5],
        "pnl": [0.1, -0.1, 0.1, -0.1, 0.1]
    })
    
    stats = calculate_trade_stats(trades)
    
    assert stats["total_trades"] == 5
    assert stats["win_rate"] == 0.6  # 3 wins out of 5
    assert stats["avg_win"] == 0.1
    assert stats["avg_loss"] == -0.1
    assert stats["profit_factor"] == 1.5  # (0.3 / 0.2) 