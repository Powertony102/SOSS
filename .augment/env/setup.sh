#!/bin/bash

# Medical Image Segmentation Project Setup Script
# This script sets up the environment for SOSS (Second Order Semi-Supervised Segmentation)

set -e  # Exit on any error

echo "=== Setting up Medical Image Segmentation Environment ==="

# Update system packages
sudo apt-get update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3 -m venv /tmp/soss_env
source /tmp/soss_env/bin/activate

# Add virtual environment activation to profile
echo "# SOSS Environment" >> $HOME/.profile
echo "source /tmp/soss_env/bin/activate" >> $HOME/.profile

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version for compatibility)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core scientific computing packages
echo "Installing scientific computing packages..."
pip install \
    numpy \
    scipy \
    scikit-image \
    scikit-learn \
    matplotlib \
    opencv-python

# Install medical imaging packages
echo "Installing medical imaging packages..."
pip install \
    h5py \
    medpy \
    SimpleITK \
    nibabel \
    pydicom \
    pynrrd

# Install machine learning and experiment tracking packages
echo "Installing ML and tracking packages..."
pip install \
    tensorboardX \
    wandb \
    tqdm \
    Pillow

# Install additional dependencies
echo "Installing additional dependencies..."
pip install \
    argparse \
    logging \
    glob2 \
    pathlib

# Verify installations
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import h5py; print(f'h5py version: {h5py.__version__}')"
python -c "import medpy; print('medpy installed successfully')"
python -c "import wandb; print('wandb installed successfully')"

# Create necessary directories
echo "Creating project directories..."
mkdir -p model
mkdir -p data
mkdir -p logs

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)\"" >> $HOME/.profile

echo "=== Setup completed successfully ==="
echo "Virtual environment activated and configured"
echo "All dependencies installed"