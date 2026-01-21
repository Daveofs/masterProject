# Master Project
AI-Enhanced Fast Cosmological Simulations for Accurate Large-Scale Structure Inference

## DISCO-DJ Setup on macOS (VS Code + Jupyter)

This guide explains how to set up **DISCO-DJ**, a cosmological N-body simulation code, on macOS using **Homebrew Python**, **virtual environments**, and **VS Code**.

### 1. Prerequisites

- **Homebrew**: [https://brew.sh](https://brew.sh)  
- **VS Code**: [https://code.visualstudio.com](https://code.visualstudio.com)  
- Optional: **Conda** installed, but **base must be deactivated**. -> conda deactivate

### 2. Create virtual environment
Run following commands in the terminal
- cd <project_folder>
- /opt/homebrew/bin/python3.12 -m venv <environment_name>
- source <environment_name>/bin/activate

### 3. Install packages
- python -m pip install jupyter ipykernel
- brew install python@3.12 gsl ninja cmake
- python -m pip install git+https://github.com/cosmo-sims/DISCO-DJ.git
