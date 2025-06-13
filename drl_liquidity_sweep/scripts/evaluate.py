"""Evaluation script for trained DRL models."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from ..data.loader import TickDataLoader
from ..env.liquidity_env import LiquiditySweepEnv
from ..utils.metrics import (calculate_mar_ratio, calculate_sharpe_ratio,
                           calculate_sortino_ratio, calculate_trade_stats)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    config_path: str,
    output_dir: Optional[str] = None,
    vec_normalize_path: Optional[str] = None,
):
    """Evaluate a trained model on test data.
    
    Args:
        model_path: Path to saved model
        config_path: Path to config YAML
        output_dir: Directory to save results
        vec_normalize_path: Path to saved VecNormalize stats
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Load test data
    loader = TickDataLoader(
        cache_dir=config["data"]["cache_dir"],
        resample_seconds=config["data"].get("resample_seconds")
    )
    
    test_data = []
    for symbol, csv_path in config["data"]["symbols"].items():
        df = loader.fetch_symbol(symbol, csv_path)
        mask = (df["time"] >= config["train"]["test_start"]) & \
               (df["time"] < config["train"]["test_end"])
        test_data.append(df[mask].assign(symbol=symbol))
    
    test_data = pd.concat(test_data)
    test_data = test_data.sort_values("time")
    
    # Create test environment
    env = LiquiditySweepEnv(
        data=test_data,
        **{k: v for k, v in config["env"].items() if k != "symbols"}
    )
    
    if vec_normalize_path:
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        
    # Load model
    model = PPO.load(model_path)
    
    # Run evaluation episode
    obs, info = env.reset()
    done = False
    
    equity_curve = [1.0]
    trades = []
    current_trade = None
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        equity_curve.append(info["equity"])
        
        # Record trades
        if info["position"] != 0 and current_trade is None:
            current_trade = {
                "entry_time": test_data.iloc[env.current_step]["time"],
                "entry_price": info["entry_price"],
                "position": info["position"],
                "symbol": test_data.iloc[env.current_step]["symbol"]
            }
        elif info["position"] == 0 and current_trade is not None:
            current_trade.update({
                "exit_time": test_data.iloc[env.current_step]["time"],
                "exit_price": test_data.iloc[env.current_step]["bid" if current_trade["position"] == 1 else "ask"],
                "pnl": (info["equity"] - equity_curve[-2]) / env.risk_unit_pct
            })
            trades.append(current_trade)
            current_trade = None
            
        done = terminated or truncated
        
    # Calculate metrics
    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    metrics = {
        "total_return": float(equity_curve[-1] - 1.0),
        "sharpe_ratio": float(calculate_sharpe_ratio(returns)),
        "sortino_ratio": float(calculate_sortino_ratio(returns)),
        "mar_ratio": float(calculate_mar_ratio(returns)),
    }
    
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        metrics.update(calculate_trade_stats(trades_df))
        
        # Calculate per-symbol attribution
        attribution = trades_df.groupby("symbol")["pnl"].agg([
            "count", "sum", "mean", "std"
        ]).round(4).to_dict("index")
        metrics["attribution"] = attribution
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save metrics
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        # Save equity curve
        pd.DataFrame({
            "time": test_data["time"].iloc[:len(equity_curve)],
            "equity": equity_curve
        }).to_csv(output_dir / "equity.csv", index=False)
        
        # Save trades
        if len(trades_df) > 0:
            trades_df.to_csv(output_dir / "trades.csv", index=False)
            
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       help="Path to saved model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config YAML")
    parser.add_argument("--output-dir", type=str,
                       help="Directory to save results")
    parser.add_argument("--vec-normalize", type=str,
                       help="Path to saved VecNormalize stats")
    args = parser.parse_args()
    
    evaluate_model(
        args.model,
        args.config,
        args.output_dir,
        args.vec_normalize
    ) 