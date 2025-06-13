"""Tests for reward calculation utilities."""

import numpy as np
import pytest

from drl_liquidity_sweep.utils.rewards import (calculate_drawdown,
                                             calculate_pnl,
                                             calculate_reward)


def test_calculate_pnl():
    """Test P&L calculation for different positions."""
    # Long position profit
    assert calculate_pnl(1, 1.0, 1.1, 1.11, 0.0) == 0.1
    
    # Short position profit
    assert calculate_pnl(-1, 1.1, 0.99, 1.0, 0.0) == 0.1
    
    # Long position loss
    assert calculate_pnl(1, 1.1, 1.0, 1.01, 0.0) == -0.1
    
    # Short position loss
    assert calculate_pnl(-1, 1.0, 1.09, 1.1, 0.0) == -0.1
    
    # Flat position
    assert calculate_pnl(0, 1.0, 1.1, 1.11, 0.0) == 0.0
    
    # With commission
    assert calculate_pnl(1, 1.0, 1.1, 1.11, 0.01) == 0.09


def test_calculate_drawdown():
    """Test drawdown calculation."""
    equity = np.array([1.0, 1.1, 1.05, 1.15, 1.0])
    max_dd, current_dd = calculate_drawdown(equity)
    
    # Maximum drawdown should be -0.13043 (from 1.15 to 1.0)
    np.testing.assert_almost_equal(max_dd, -0.13043, decimal=5)
    
    # Current drawdown should be -0.13043 (at final point)
    np.testing.assert_almost_equal(current_dd, -0.13043, decimal=5)
    
    # Test with constant equity
    equity = np.array([1.0, 1.0, 1.0])
    max_dd, current_dd = calculate_drawdown(equity)
    assert max_dd == 0.0
    assert current_dd == 0.0
    
    # Test with monotonic increase
    equity = np.array([1.0, 1.1, 1.2])
    max_dd, current_dd = calculate_drawdown(equity)
    assert max_dd == 0.0
    assert current_dd == 0.0


def test_calculate_reward():
    """Test reward calculation with drawdown penalty."""
    # Positive P&L, no drawdown
    assert calculate_reward(0.1, 0.0, 1.0) == 0.1
    
    # Positive P&L with drawdown
    assert calculate_reward(0.1, -0.05, 1.0) == 0.15
    
    # Negative P&L with drawdown
    assert calculate_reward(-0.1, -0.05, 1.0) == -0.05
    
    # Different lambda penalty
    assert calculate_reward(0.1, -0.05, 2.0) == 0.2 