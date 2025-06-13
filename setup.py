"""Setup script for drl-liquidity-sweep."""

from setuptools import find_packages, setup

setup(
    name="drl-liquidity-sweep",
    version="0.1.0",
    description="Deep Reinforcement Learning for Market Making",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.2.0",
        "gymnasium==0.29.1",
        "stable-baselines3==2.3.0",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "pyyaml==6.0.1",
        "plotly==5.18.0",
        "matplotlib==3.8.2",
        "ta==0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.3",
            "black==23.12.1",
            "isort==5.13.2",
            "flake8==7.0.0",
            "mypy==1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "drl-train=drl_liquidity_sweep.scripts.train:main",
            "drl-evaluate=drl_liquidity_sweep.scripts.evaluate:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 