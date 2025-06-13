"""Tests for the LiquiditySweepEnv."""

import numpy as np
import pandas as pd
import pytest

from drl_liquidity_sweep.env.liquidity_env import LiquiditySweepEnv
from drl_liquidity_sweep.utils.rewards import calculate_pnl


@pytest.fixture
def sample_data():
    """Create sample tick data for testing."""
    times = pd.date_range("2024-01-01", periods=100, freq="S")
    data = pd.DataFrame({
        "time": times,
        "bid": np.linspace(1.0, 1.1, 100),
        "ask": np.linspace(1.001, 1.101, 100),
        "volume": np.ones(100),
    })
    data["mid"] = (data["bid"] + data["ask"]) / 2
    data["spread"] = data["ask"] - data["bid"]
    return data


def test_env_init(sample_data):
    """Test environment initialization."""
    env = LiquiditySweepEnv(sample_data)
    assert env.action_space.n == 3
    assert env.observation_space.shape == (10,)


def test_env_reset(sample_data):
    """Test environment reset."""
    env = LiquiditySweepEnv(sample_data)
    obs, info = env.reset()
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (10,)
    assert info["equity"] == 1.0
    assert info["position"] == 0


def test_env_step(sample_data):
    """Test environment step with known P&L."""
    env = LiquiditySweepEnv(sample_data, commission=0.0)
    env.reset()
    
    # Take long position
    obs, reward, terminated, truncated, info = env.step(1)
    entry_price = sample_data.iloc[0]["ask"]
    
    # Close position
    obs, reward, terminated, truncated, info = env.step(0)
    exit_price = sample_data.iloc[1]["bid"]
    
    expected_pnl = calculate_pnl(1, entry_price, exit_price, exit_price)
    actual_pnl = (info["equity"] - 1.0) / env.risk_unit_pct
    
    np.testing.assert_almost_equal(actual_pnl, expected_pnl)


def test_env_daily_loss_limit(sample_data):
    """Test environment termination on daily loss limit."""
    env = LiquiditySweepEnv(
        sample_data,
        commission=0.0,
        risk_unit_pct=1.0,
        daily_loss_limit_pct=0.01
    )
    env.reset()
    
    terminated = False
    while not terminated:
        # Take alternating positions to generate losses
        action = env.action_space.sample()
        _, _, terminated, _, info = env.step(action)
        if info["equity"] < 0.99:  # Should trigger daily loss limit
            assert terminated 