#!/bin/bash
#abc
set -e

echo "==== 1. Installing essential system packages ===="
apt-get update
apt-get install -y build-essential curl git bzip2 wget
apt install nvtop



echo "==== 2. Checking if Anaconda is installed in /workspace ===="
INSTALLER="/workspace/Anaconda3-2025.06-0-Linux-x86_64.sh"
INSTALL_DIR="/workspace/anaconda3"
BASHRC="$HOME/.bashrc"

# 1. Download installer if needed and run it
if [ -f "$INSTALLER" ]; then
    echo "File $INSTALLER already exists."
else
    echo "File $INSTALLER does not exist. Downloading and running installer..."
    wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh -P /workspace
    bash /workspace/Anaconda3-2025.06-0-Linux-x86_64.sh -b -p "$INSTALL_DIR"
fi

# 2. Add conda to PATH if not already present
if ! grep -q '/workspace/anaconda3/bin' "$BASHRC"; then
    echo "Adding conda to PATH in $BASHRC..."
    echo 'export PATH="/workspace/anaconda3/bin:$PATH"' >> "$BASHRC"
fi

# 3. Also add to current session for immediate use
export PATH="/workspace/anaconda3/bin:$PATH"

echo "Conda installation complete. Run 'source ~/.bashrc' or start a new shell to activate conda permanently."


echo "==== 5. Checking if 'ml' Conda environment exists ========================"

if conda env list | grep -qE '(^|\s)ml($|\s)'; then
  echo "'ml' Conda environment already exists. Skipping creation and setup."
else
  echo "'ml' Conda environment not found. Creating and setting up."
  conda config --set solver libmamba
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

  conda create -y -n ml python=3.12.9

  echo "==== 6. Activating the ml environment ===="
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ml
  
  pip install --upgrade pip
  conda install -n ml ipykernel --update-deps --force-reinstall
fi
  
echo "==== 7. Installing local flashml package in editable mode ===="

if [ ! -d "/workspace/flashml" ]; then
  git clone https://github.com/smtmRadu/flashml flashml
  cd /workspace/flashml
  pip install -e .
  echo "Directory /workspace/flashml does not exist. Skipping local package install."
fi
echo "==== All done! Your 'ml' environment is set up. ===="

echo 'export PATH="$HOME/workspace/anaconda3/bin:$PATH"' >> ~/.bashrc
echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc

conda init bash

# Make 'ml' the default environment on new terminals
if ! grep -q "conda activate ml" ~/.bashrc; then
    echo "conda activate ml" >> ~/.bashrc
fi

conda activate ml
pip install ipywidgets
pip install vllm

source ~/.bashrc

python -m pip install --upgrade pip
echo "==== Setup completed successfully! ===="
echo "RESTART this terminal"