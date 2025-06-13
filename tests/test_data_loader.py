"""Tests for the TickDataLoader."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from drl_liquidity_sweep.data.loader import TickDataLoader


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file with tick data."""
    csv_path = tmp_path / "test.csv"
    data = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=100, freq="S"),
        "bid": np.linspace(1.0, 1.1, 100),
        "ask": np.linspace(1.001, 1.101, 100),
        "volume": np.ones(100),
    })
    data.to_csv(csv_path, index=False, header=False)
    return csv_path


def test_load_tick_csv(sample_csv):
    """Test loading tick data from CSV."""
    loader = TickDataLoader()
    df = loader.load_tick_csv(sample_csv, "EURUSD")
    
    assert len(df) == 100
    assert all(col in df.columns for col in ["time", "bid", "ask", "volume", "mid", "spread"])
    assert (df["spread"] == df["ask"] - df["bid"]).all()
    assert (df["mid"] == (df["bid"] + df["ask"]) / 2).all()


def test_resample_bars(sample_csv):
    """Test resampling to N-second bars."""
    loader = TickDataLoader(resample_seconds=10)
    df = loader.load_tick_csv(sample_csv, "EURUSD")
    
    assert len(df) == 10  # 100 seconds / 10 seconds per bar
    assert df["volume"].sum() == 100  # Volume should be preserved


def test_fetch_symbol_caching(sample_csv, tmp_path):
    """Test data caching functionality."""
    cache_dir = tmp_path / "cache"
    loader = TickDataLoader(cache_dir=cache_dir)
    
    # First load should create cache
    df1 = loader.fetch_symbol("EURUSD", sample_csv)
    assert (cache_dir / "EURUSD.parquet").exists()
    
    # Second load should use cache
    df2 = loader.fetch_symbol("EURUSD", sample_csv)
    pd.testing.assert_frame_equal(df1, df2)
    
    # Force reload should bypass cache
    df3 = loader.fetch_symbol("EURUSD", sample_csv, force_reload=True)
    pd.testing.assert_frame_equal(df1, df3) 