# DRL Liquidity Sweep

A Deep Reinforcement Learning framework for market making and liquidity sweeping using high-frequency data.

## Features

- Load and process tick/1-second data from MT5 (bid/ask prices)
- Gymnasium-compatible environment with realistic fill simulation
- Configurable latency jitter and commission modeling
- Multi-instrument support with proper time alignment
- Stable-Baselines3 PPO implementation with curriculum learning
- Comprehensive evaluation metrics and visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/drl-liquidity-sweep.git
cd drl-liquidity-sweep
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data Preparation

1. In MetaTrader 5, export tick data for your instruments:
   - Open Market Watch
   - Right-click on the symbol
   - Select "Ticks" from the context menu
   - Choose your date range
   - Click "Export" and save as CSV

2. Place your CSV files in the `data/` directory with filenames matching the config:
```
data/
├── EURUSD.csv
└── GBPUSD.csv
```

## Configuration

The default configuration is in `drl_liquidity_sweep/config/default_config.yaml`. Key parameters:

- `data`: Data loading settings (symbols, resampling)
- `env`: Environment parameters (commission, risk limits)
- `ppo`: PPO hyperparameters
- `train`: Training schedule and logging

## Usage

1. Train a model:
```bash
python -m drl_liquidity_sweep.scripts.train --config drl_liquidity_sweep/config/default_config.yaml
```

2. Monitor training:
```bash
tensorboard --logdir logs/tensorboard
```

3. Evaluate the model:
```bash
python -m drl_liquidity_sweep.scripts.evaluate --model logs/best_model/best_model.zip --config drl_liquidity_sweep/config/default_config.yaml
```

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black drl_liquidity_sweep tests
isort drl_liquidity_sweep tests
```

## License

MIT License 