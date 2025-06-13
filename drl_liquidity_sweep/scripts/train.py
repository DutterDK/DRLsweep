"""Training script for the liquidity sweep DRL agent."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from ..data.loader import TickDataLoader
from ..env.liquidity_env import LiquiditySweepEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_env(data: pd.DataFrame, config: Dict, seed: Optional[int] = None):
    """Create a vectorized environment factory."""
    def _init():
        env = LiquiditySweepEnv(
            data=data,
            commission=config["env"]["commission"],
            lambda_penalty=config["env"]["lambda_penalty"],
            latency_jitter=config["env"].get("latency_jitter"),
            risk_unit_pct=config["env"]["risk_unit_pct"],
            daily_loss_limit_pct=config["env"]["daily_loss_limit_pct"],
        )
        env = Monitor(env)
        if seed is not None:
            env.seed(seed)
        return env
    return _init


def train(config_path: str):
    """Train the agent using configuration from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Load data
    loader = TickDataLoader(
        cache_dir=config["data"]["cache_dir"],
        resample_seconds=config["data"].get("resample_seconds")
    )
    
    train_data = []
    for symbol, csv_path in config["data"]["symbols"].items():
        df = loader.fetch_symbol(symbol, csv_path)
        mask = (df["time"] >= config["train"]["train_start"]) & \
               (df["time"] < config["train"]["train_end"])
        train_data.append(df[mask])
    
    train_data = pd.concat(train_data)
    train_data = train_data.sort_values("time")
    
    # Create vectorized environment
    env_fns = [make_env(train_data, config, seed=i) 
               for i in range(config["train"]["n_envs"])]
    
    if config["train"]["n_envs"] > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
        
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    # Create validation environment
    val_data = []
    for symbol, csv_path in config["data"]["symbols"].items():
        df = loader.fetch_symbol(symbol, csv_path)
        mask = (df["time"] >= config["train"]["val_start"]) & \
               (df["time"] < config["train"]["val_end"])
        val_data.append(df[mask])
    
    val_data = pd.concat(val_data)
    val_data = val_data.sort_values("time")
    
    val_env = Monitor(LiquiditySweepEnv(
        data=val_data,
        **{k: v for k, v in config["env"].items() if k != "symbols"}
    ))
    
    # Set up callbacks
    log_dir = Path(config["train"]["log_dir"])
    log_dir.mkdir(exist_ok=True, parents=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config["train"]["checkpoint_freq"],
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppo_liquidity"
    )
    
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "evaluations"),
        eval_freq=config["train"]["eval_freq"],
        deterministic=True,
        render=False
    )
    
    # Create and train PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
        **config["ppo"]
    )
    
    model.learn(
        total_timesteps=config["train"]["total_timesteps"],
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(str(log_dir / "final_model"))
    env.save(str(log_dir / "vec_normalize.pkl"))
    
    logger.info(f"Training complete. Models saved to {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML config file")
    args = parser.parse_args()
    
    train(args.config) 