"""Reward calculation utilities for the DRL environment."""

from typing import Tuple

import numpy as np
import pandas as pd


def calculate_pnl(
    position: int,  # -1, 0, or 1
    entry_price: float,
    current_bid: float,
    current_ask: float,
    commission: float = 0.0,
) -> float:
    """Calculate realized P&L for a position.
    
    Args:
        position: Current position (-1=short, 0=flat, 1=long)
        entry_price: Position entry price
        current_bid: Current bid price
        current_ask: Current ask price
        commission: Commission per trade
        
    Returns:
        Realized P&L
    """
    if position == 0:
        return 0.0
        
    exit_price = current_bid if position == 1 else current_ask
    pnl = position * (exit_price - entry_price) - commission
    return pnl


def calculate_drawdown(
    equity_curve: np.ndarray,
) -> Tuple[float, float]:
    """Calculate maximum drawdown and current drawdown.
    
    Args:
        equity_curve: Array of cumulative equity values
        
    Returns:
        Tuple of (max_drawdown, current_drawdown)
    """
    rolling_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - rolling_max
    max_drawdown = np.min(drawdown)
    current_drawdown = drawdown[-1]
    
    return max_drawdown, current_drawdown


def calculate_reward(
    pnl: float,
    drawdown_increment: float,
    lambda_penalty: float = 1.0,
) -> float:
    """Calculate reward as P&L minus drawdown penalty.
    
    Args:
        pnl: Realized P&L
        drawdown_increment: Change in drawdown
        lambda_penalty: Penalty coefficient for drawdown
        
    Returns:
        Reward value
    """
    reward = pnl - lambda_penalty * drawdown_increment
    return reward 